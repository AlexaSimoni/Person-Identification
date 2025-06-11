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
    add_clip_reference, is_unique_against_recent_clip_refs,
)
from server.Utils import framesGlobals

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
        logger.info(f"[FlowTracker] Using flow_net type: {type(self.flow_net).__name__}")
    """
    def compute_iou(self, boxa, boxb):
        x_a = max(boxa[0], boxb[0])
        y_a = max(boxa[1], boxb[1])
        x_b = min(boxa[0] + boxa[2], boxb[0] + boxb[2])
        y_b = min(boxa[1] + boxa[3], boxb[1] + boxb[3])

        interarea = max(0, x_b - x_a) * max(0, y_b - y_a)
        boxaarea = boxa[2] * boxa[3]
        boxbarea = boxb[2] * boxb[3]

        iou = interarea / float(boxaarea + boxbarea - interarea + 1e-5)
        return iou
    """
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

        """
        # Retrieve previous and current frames
        prev_frame = all_even_frames[self.last_frame_index]
        curr_frame = all_even_frames[current_frame_index]
        if prev_frame is None or curr_frame is None:
            return self.last_box
            """

        # Apply optical flow to update bounding box
        #updated_box = self.update_bbox_with_optical_flow(prev_frame, curr_frame, self.last_box, current_frame_index)
        updated_box = self.update_bbox_with_optical_flow(self.last_frame_index, current_frame_index, self.last_box)
        logger.info(f"[FlowNet] Updated box via optical flow")
        if updated_box is not None:
            self.last_box = updated_box
            self.last_frame_index = current_frame_index

            # Extract image crop from updated box
            x, y, w, h = updated_box
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
            logger.info(
                f"[CLIP] Similarity to reference for UUID {self.uuid} at frame {current_frame_index}: {max_similarity:.2f}")

            # similarity = get_best_clip_similarity(self.uuid, clip_emb)
            # logger.info(
            #   f"[CLIP] Similarity to reference for UUID {self.uuid} at frame {current_frame_index}: {similarity:.2f}")
            # Check if appearance is valid for storage

            # If similar enough and unique, store this detection
            if max_similarity >= CLIP_STORE_THRESHOLD:
                if is_unique_against_recent_clip_refs(self.uuid, clip_emb, threshold=0.05):
                    #self.store_flow_detection(curr_frame, updated_box, current_frame_index)
                    self.store_flow_detection(current_frame_index, updated_box)
                    logger.info(f"[FlowNet] Stored unique flow detection with sim={max_similarity:.2f}")
                else:
                    logger.info(f"[FlowNet] Rejected — not unique against recent crops (sim={max_similarity:.2f})")

            # Update CLIP reference memory if similarity is high
            if max_similarity >= CLIP_UPDATE_THRESHOLD:
                add_clip_reference(self.uuid, clip_emb, similarity=max_similarity)
                logger.info(f"[CLIP] Updated reference list for UUID {self.uuid} (new sim={max_similarity:.2f})")

        return self.last_box

    # Update bounding box using optical flow between two frames
    # Inputs: prev_frame (image), next_frame (image), bbox - (x, y, w, h), current_frame_index
    # Output: updated bounding box (x, y, w, h) if motion detected; otherwise original box
    #def update_bbox_with_optical_flow(self, prev_frame: np.ndarray, next_frame: np.ndarray, bbox: tuple,
    #                                 current_frame_index: int):
    #                               min_motion_threshold: float = 1.0) -> tuple:




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
        # x, y, w, h = bbox
        #margin_ratio = 0.5
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
        w = int(w * scale_x)
        h = int(h * scale_y)

       # Extract region of interest in the flow map
        flow_roi = flow[rel_y:rel_y + h, rel_x:rel_x + w]
        flow_x = flow_roi[..., 0]
        flow_y = flow_roi[..., 1]
        magnitude_map = np.sqrt(flow_x ** 2 + flow_y ** 2)

        # Focus on top 30% motion (mask top 30% of high-motion pixels)
        threshold = np.percentile(magnitude_map, 70)
        # Focus on top 50% motion
        # threshold = np.percentile(magnitude_map, 50)
        mask = magnitude_map >= threshold

        # Abort if not enough motion
        # if np.count_nonzero(mask) < 20
        if np.count_nonzero(mask) < 10:
            logger.info(f"[FlowNet] UUID: {self.uuid} | Insufficient motion — keeping box")
            self.frames_since_last_match += 1
            return bbox

        # Compute weighted average displacement (average of dx, dy)
        weights = magnitude_map[mask]
        dx = np.sum(flow_x[mask] * weights) / np.sum(weights)
        dy = np.sum(flow_y[mask] * weights) / np.sum(weights)
        magnitude = np.sqrt(dx ** 2 + dy ** 2)

        logger.info(f"[FlowNet] UUID: {self.uuid} | dx: {dx:.2f}, dy: {dy:.2f}, mag: {magnitude:.2f}")
        # respond to smaller motion
        min_motion_threshold = 1.0
        if magnitude < min_motion_threshold:
            self.frames_since_last_match += 1
            return bbox
        else:
            self.frames_since_last_match = 0

        # Apply the flow displacement (movement)
        x_new = int(x + dx / scale_x)
        y_new = int(y + dy / scale_y)
        # Estimate scale based on motion variability (motion spread)
        std_x = np.std(flow_x[mask])
        std_y = np.std(flow_y[mask])
        scale_factor = 1.0 + 2.5 * ((std_x + std_y) / 10.0)
        # scale_factor = 1.0 + 1.5 * ((std_x + std_y) / 5.0)

        # Clamp scale to never shrink under 90%, never grow beyond 120%
        # scale_factor = np.clip(scale_factor, 0.9, 1.5)
        scale_factor = np.clip(scale_factor, 0.4, 1.2)

        # Resize box and clamp boundaries
        roi_w = flow_roi.shape[1]
        roi_h = flow_roi.shape[0]

        new_w = int(roi_w / scale_x * scale_factor)
        new_h = int(roi_h / scale_y * scale_factor)

        new_w = max(10, min(new_w, w_frame - x_new))
        new_h = max(10, min(new_h, h_frame - y_new))
        x_new = max(0, min(x_new, w_frame - new_w))
        y_new = max(0, min(y_new, h_frame - new_h))
        # Save debug crops for test
        #cv2.imwrite(f"flow_debug_prev_{self.uuid}.jpg", prev_crop)
        #cv2.imwrite(f"flow_debug_next_{self.uuid}.jpg", next_crop)
        logger.info(f"[FlowNet] Updated box → x={x_new}, y={y_new}, w={new_w}, h={new_h}, scale={scale_factor:.2f}")
        return x_new, y_new, new_w, new_h

        # return x_new, y_new, w, h

    # Saves the cropped image of the current detection to global memory (for later use or DB)
    # Inputs: frame (image), box - (x, y, w, h), frame_index - index of frame
    # Output: None (stores image in framesGlobals.flowdetected_frames)
    #def store_flow_detection(self, frame: np.ndarray, box: tuple, frame_index: int):

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


