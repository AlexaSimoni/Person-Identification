import numpy as np
import logging
import cv2
import time
import base64

from server.Utils.framesGlobals import all_even_frames, detections_frames, dir_path, flow_clip_reference
from server.FlowNet_Component.Cropper import Cropper
from server.FlowNet_Component.clip_utils import (
    get_clip_embedding,
    compare_clip_embeddings,
    add_clip_reference, is_unique_against_recent_clip_refs
)
from server.Utils import framesGlobals
from server.config.config import USE_CLIP_IN_FLOWTRACKING

logger = logging.getLogger(__name__)

# FlowTracker class for tracking a person identified by UUID across video frames using optical flow
# Stores last known bounding box and frame
# Incorporates CLIP-based appearance verification for reliability
# Inputs: flow_net (optical flow model used FlowNetSWrapper or SimpleFlowNet), uuid (person id)
class FlowTracker:
    def __init__(self, flow_net, uuid):
        self.initial_facenet_box = None
        self.flow_net = flow_net  # FlowNetSWrapper or SimpleFlowNet
        self.uuid = uuid  # Person UUID
        self.last_frame_index = None  # Last known frame
        self.last_box = None  # Last known box (x, y, w, h)
        self.best_score = 0.0  # Best similarity score from FaceNet
        self.frames_since_last_match = 0  # Counter for tracking without FaceNet
        self.use_clip = USE_CLIP_IN_FLOWTRACKING  # Flag from config.py
        logger.info(f"[FlowTracker] Using flow_net type: {type(self.flow_net).__name__}")

    # Used to update the tracked bounding box to the current frame using optical flow and CLIP embedding validation
    # Inputs: current_frame_index - index of the new frame to track
    # Output: Updated bounding box (x, y, w, h) if successful, otherwise returns last known box
    def update_track_frame(self, current_frame_index):
        #from server.FlowNet_Component import FlowNet_Utils
        max_wait_attempts = 5
        wait_count = 0
        CLIP_STORE_THRESHOLD = 0.75 # Minimum similarity for CLIP fallback acceptance
        CLIP_UPDATE_THRESHOLD = 0.85    # Similarity required to update CLIP references

        # Wait until both frames are available in shared memory
        while (
                self.last_frame_index not in all_even_frames or current_frame_index not in all_even_frames
        ) and wait_count < max_wait_attempts:
            logger.info(
                f"[FlowTracker] Waiting for frames... attempt {wait_count + 1}/5 | last={self.last_frame_index}, current={current_frame_index}"
            )
            time.sleep(0.05)
            wait_count += 1

        # Abort if initialization hasn't happened
        if self.last_frame_index is None or self.last_box is None:
            logger.warning(f"[FlowNet] Skipping tracking for UUID {self.uuid} — no FaceNet match yet.")
            return None
        # Abort if frames are still missing
        if self.last_frame_index not in all_even_frames or current_frame_index not in all_even_frames:
            logger.warning(f"[FlowTracker] Skipping frame {current_frame_index} after waiting — still not found")
            return self.last_box

        # Apply optical flow to update bounding box
        #updated_box = self.update_bbox_with_optical_flow(prev_frame, curr_frame, self.last_box, current_frame_index)
        updated_box = self.update_bbox_with_optical_flow(self.last_frame_index, current_frame_index, self.last_box)
        logger.info(f"[FlowNet] Updated box via optical flow")
        if updated_box is not None:
            self.last_box = updated_box
            self.last_frame_index = current_frame_index

            # Extract image crop from updated box
            x, y, w, h = updated_box
            if self.use_clip:
            # Extract crop and compute CLIP embedding
                curr_frame = all_even_frames[current_frame_index]
                crop = curr_frame[y:y + h, x:x + w]
                clip_emb = get_clip_embedding(crop)
                if clip_emb is None:
                    logger.warning(f"[FlowNet] Failed to extract CLIP embedding at frame {current_frame_index}")
                    return self.last_box

                # Compare new embedding to existing CLIP reference embeddings
                refs = flow_clip_reference.get(self.uuid, {}).get("clip_embeddings", [])
                similarities = [compare_clip_embeddings(clip_emb, ref) for ref in refs]
                max_similarity = max(similarities, default=0.0)
                logger.info(f"[CLIP] Similarity to reference for UUID {self.uuid} at frame {current_frame_index}: {max_similarity:.2f}")

                # If similar enough and unique, store this detection
                if max_similarity >= CLIP_STORE_THRESHOLD:
                    if is_unique_against_recent_clip_refs(self.uuid, clip_emb, threshold=0.05):
                        self.store_flow_detection(current_frame_index, updated_box)
                        logger.info(f"[FlowNet] Stored unique flow detection with sim={max_similarity:.2f}")
                    else:
                        logger.info(f"[FlowNet] Rejected — not unique against recent crops (sim={max_similarity:.2f})")

                # Update CLIP reference memory if similarity is high
                if max_similarity >= CLIP_UPDATE_THRESHOLD:
                    add_clip_reference(self.uuid, clip_emb, similarity=max_similarity)
                    logger.info(f"[CLIP] Updated reference list for UUID {self.uuid} (new sim={max_similarity:.2f})")

            else:
                # Skip CLIP logic, store directly
                self.store_flow_detection(current_frame_index, updated_box)
                logger.info(f"[FlowNet] Stored flow detection without CLIP filtering (CLIP disabled)")

        return self.last_box

    # Update bounding box using optical flow between two frames
    # Inputs: prev_frame_index, current_frame_index - frame indices; bbox - last known (x, y, w, h)
    # Output: updated bounding box (x, y, w, h) if motion detected; otherwise original box
    def update_bbox_with_optical_flow(self, prev_frame_index: int, current_frame_index: int, bbox: tuple):
        # Retrieve previous and current frames
        prev_frame = all_even_frames.get(prev_frame_index)
        curr_frame = all_even_frames.get(current_frame_index)

        if prev_frame is None or curr_frame is None:
            logger.warning(f"[FlowNet] Missing frames at indices {prev_frame_index}, {current_frame_index}")
            return bbox  # fallback to last known box
        # Use optical flow to update the bounding box between two frames

        h_frame, w_frame = prev_frame.shape[:2]

        margin_ratio = 1.0
        # Crop areas around the bbox with margin
        prev_crop, (x1, y1) = Cropper.crop_with_margin(prev_frame, bbox, margin_ratio)
        next_crop, _ = Cropper.crop_with_margin(curr_frame, bbox, margin_ratio)

        # Compute optical flow between cropped regions
        logger.info(f"[FlowTracker] Calling compute_flow() for UUID {self.uuid}")
        flow = self.flow_net.compute_flow(prev_crop, next_crop, debug_uuid=self.uuid,
                                          frame_index=current_frame_index)
        logger.info(f"[FlowTracker] compute_flow() returned for UUID {self.uuid}")

        x, y, w, h = bbox

        # Scale coordinates to flow map resolution
        flow_h, flow_w = flow.shape[:2]
        scale_x = flow_w / prev_crop.shape[1]
        scale_y = flow_h / prev_crop.shape[0]

        rel_x = int((x - x1) * scale_x)
        rel_y = int((y - y1) * scale_y)
        w_scaled = int(w * scale_x)
        h_scaled = int(h * scale_y)

        # Guard for bad coords
        rel_x = max(0, min(rel_x, flow_w - 1))
        rel_y = max(0, min(rel_y, flow_h - 1))
        w_scaled = max(1, min(w_scaled, flow_w - rel_x))
        h_scaled = max(1, min(h_scaled, flow_h - rel_y))

        # Extract region of interest in the flow map
        flow_roi = flow[rel_y:rel_y + h_scaled, rel_x:rel_x + w_scaled]

        flow_x = flow_roi[..., 0]
        flow_y = flow_roi[..., 1]
        magnitude_map = np.sqrt(flow_x ** 2 + flow_y ** 2)

        logger.info(f"[FlowNet] UUID: {self.uuid} | Flow ROI shape: {flow_roi.shape} | Mean mag: {np.mean(magnitude_map):.2f} | Non-zero flow: {np.count_nonzero(magnitude_map):d}")

        threshold = np.percentile(magnitude_map, 70)
        mask = magnitude_map >= threshold

        if np.count_nonzero(mask) < 10:
            logger.info(f"[FlowNet] UUID: {self.uuid} | Low motion — freezing box.")
            self.frames_since_last_match += 1
            return bbox

        self.frames_since_last_match = 0

        # center-biased, top-k strongest selection
        H, W = magnitude_map.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W]
        # Gaussian centered in the middle of the bbox to focus on the person
        sx, sy = 0.30 * W, 0.30 * H  # 30% of width/height as std
        gauss = np.exp(-(((xx - W / 2) ** 2) / (2 * sx * sx) + ((yy - H / 2) ** 2) / (2 * sy * sy)))

        #weights = magnitude_map[mask]
        # Combine motion strength and center prior
        weighted = magnitude_map * gauss

        # Take the strongest K% pixels (default 15%)
        TOPK_FRAC = 0.15
        #TOPK_FRAC = 0.5
        #TOPK_FRAC = 0.2
        k = max(10, int(weighted.size * TOPK_FRAC))
        flat_idx = np.argpartition(weighted.ravel(), -k)[-k:]
        sel_w = weighted.ravel()[flat_idx]
        sel_fx = flow_x.ravel()[flat_idx]
        sel_fy = flow_y.ravel()[flat_idx]

        if sel_w.sum() == 0 or k == 0:
            logger.info(f"[FlowNet] UUID: {self.uuid} | No strong motion — freezing box.")
            return bbox

        #dx = np.sum(flow_x[mask] * weights) / np.sum(weights)
        #dy = np.sum(flow_y[mask] * weights) / np.sum(weights)
        # Weighted average displacement (robust to background)
        dx = float(np.sum(sel_fx * sel_w) / np.sum(sel_w))
        dy = float(np.sum(sel_fy * sel_w) / np.sum(sel_w))
        magnitude = np.sqrt(dx ** 2 + dy ** 2)

        logger.info(f"[FlowNet] UUID: {self.uuid} | dx: {dx:.2f}, dy: {dy:.2f}, mag: {magnitude:.2f}")

        # Be more sensitive to small motions
        #min_motion_threshold = 0.4
        min_motion_threshold = 0.1

        if magnitude < min_motion_threshold:
            self.frames_since_last_match += 1
            return bbox
        else:
            self.frames_since_last_match = 0

        #x_new = int(round(x + dx / scale_x))
        #y_new = int(round(y + dy / scale_y))

        # move more decisively, barely resize
        #MOTION_GAIN = 1.25  # amplify dx,dy to follow the person more aggressively
        #MOTION_GAIN = 2.0  # amplify dx,dy to follow the person more aggressively
        MOTION_GAIN = 5.0
        x_new = int(round(x + MOTION_GAIN * (dx / scale_x)))
        y_new = int(round(y + MOTION_GAIN * (dy / scale_y)))

        # Very small / no resize
        NO_RESIZE = False  # set to True to lock size completely
        MAX_SCALE_DELTA = 0.02  # ±2%

        if NO_RESIZE:
            new_w, new_h = w, h
        else:
            std_x = np.std(sel_fx)
            std_y = np.std(sel_fy)
            #scale_delta = (std_x + std_y) / 30.0  # heavily dampen
            #scale_factor = 1.0 + np.clip(scale_delta, -MAX_SCALE_DELTA, MAX_SCALE_DELTA)
            #new_w = int(w * scale_factor)
            #new_h = int(h * scale_factor)
            # Width can resize slightly (±5%)
            scale_x_factor = 1.0 + np.clip(std_x / 20.0, -0.05, 0.05)

            # Height is mostly locked (±2% max)
            scale_y_factor = 1.0 + np.clip(std_y / 100.0, -0.01, 0.01)

            new_w = int(w * scale_x_factor)
            new_h = int(h * scale_y_factor)
        #std_x = np.std(flow_x[mask])
        #std_y = np.std(flow_y[mask])
        #scale_delta = ((std_x + std_y) / 10.0)
        #if scale_delta > 0.01:
        #    scale_factor = 1.0 + min(0.4, scale_delta * 0.8)
        #else:
        #    scale_factor = 0.95  # slight shrink when motion is stable
        #scale_factor = np.clip(scale_factor, 0.9, 1.15)

        #new_w = int(w * scale_factor)
        #new_h = int(h * scale_factor)

        #new_w = max(10, min(new_w, w_frame - x_new))
        #new_h = max(10, min(new_h, h_frame - y_new))
        # Clamp to frame bounds
        new_w = max(10, min(new_w, w_frame - 1))
        new_h = max(10, min(new_h, h_frame - 1))
        x_new = max(0, min(x_new, w_frame - new_w))
        y_new = max(0, min(y_new, h_frame - new_h))

        #logger.info(f"[FlowNet] Updated box → x={x_new}, y={y_new}, w={new_w}, h={new_h}, scale={scale_factor:.2f}")
        return x_new, y_new, new_w, new_h

    # Stores cropped image of updated bounding box into memory (for possible later use or DB upload)
    # Inputs: frame_index - frame ID to retrieve from shared memory; box - (x, y, w, h) bounding box
    # Output: None (stores cropped image in framesGlobals.flowdetected_frames)
    def store_flow_detection(self, frame_index: int, box: tuple):
        frame = all_even_frames.get(frame_index)
        if frame is None:
            logger.warning(f"[FlowNet] Cannot store crop — frame {frame_index} not available")
            return

        # Clamp bounding box coordinates within frame dimensions
        x, y, w, h = box
        h_frame, w_frame = frame.shape[:2]
        x, y = max(0, x), max(0, y)
        x2, y2 = min(x + w, w_frame), min(y + h, h_frame)
        cropped = frame[y:y2, x:x2]

        if cropped.size == 0:
            logger.warning(f"[FlowNet] Skipping empty crop for memory storage at frame {frame_index}")
            return

        # Encode crop to JPEG and convert to base64
        success, buffer = cv2.imencode('.jpg', cropped)
        if not success:
            logger.warning(f"[FlowNet] Failed to encode crop at frame {frame_index}")
            return

        base64_crop = base64.b64encode(buffer).decode('utf-8')
        # Save in global frame store
        framesGlobals.flowdetected_frames[frame_index] = {
            "cropped_image": base64_crop,
            "similarity": 0  # placeholder; update later if needed
        }

        logger.info(f"[FlowNet] Stored FlowNet detection in memory for UUID {self.uuid} at frame {frame_index}")


