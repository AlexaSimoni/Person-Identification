import numpy as np
import logging
import cv2
import time
from server.Utils.framesGlobals import all_even_frames, detections_frames, dir_path
from server.FlowNet_Component.Cropper import Cropper
from server.config.config import SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

class FlowTracker:
    def __init__(self, flow_net, uuid):
        self.flow_net = flow_net    #FlowNetSWrapper or SimpleFlowNet
        self.uuid = uuid    #Person UUID
        self.last_frame_index = None    #Last known frame
        self.last_box = None    #Last known box (x, y, w, h)
        self.best_score = 0.0   #Best similarity score from FaceNet
        self.frames_since_last_match = 0  # Counter for tracking without FaceNet
        logger.info(f"[FlowTracker] Using flow_net type: {type(self.flow_net).__name__}")

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

    def update_track_frame(self, current_frame_index):
        max_wait_attempts = 5
        wait_count = 0
        while (
                self.last_frame_index not in all_even_frames or current_frame_index not in all_even_frames) and wait_count < max_wait_attempts:
            logger.info(
                f"[FlowTracker] Waiting for frames... attempt {wait_count + 1}/5 | last={self.last_frame_index}, current={current_frame_index}")
            time.sleep(0.05)  # wait 50ms
            wait_count += 1

        if self.last_frame_index not in all_even_frames or current_frame_index not in all_even_frames:
            logger.warning(f"[FlowTracker] Skipping frame {current_frame_index} after waiting — still not found")
            logger.info(f"[FlowTracker] all_even_frames keys: {sorted(all_even_frames.keys())}")

            return self.last_box

        prev_frame = all_even_frames[self.last_frame_index]
        curr_frame = all_even_frames[current_frame_index]

        #Update box using optical flow
        #updated_box = self.update_bbox_with_optical_flow(prev_frame, curr_frame, self.last_box)
        updated_box = self.update_bbox_with_optical_flow(prev_frame, curr_frame, self.last_box, current_frame_index)

        if updated_box is not None:
            self.last_box = updated_box
            self.last_frame_index = current_frame_index

        return self.last_box

    def update_bbox_with_optical_flow(self, prev_frame: np.ndarray, next_frame: np.ndarray, bbox: tuple,
                                      current_frame_index: int,
                                      min_motion_threshold: float = 1.0) -> tuple:
        #Use optical flow to update the bounding box between two frames

        h_frame, w_frame = prev_frame.shape[:2]
        #x, y, w, h = bbox
        margin_ratio = 0.75
        #Crop both frames with margin
        prev_crop, (x1, y1) = Cropper.crop_with_margin(prev_frame, bbox, margin_ratio)
        next_crop, _ = Cropper.crop_with_margin(next_frame, bbox, margin_ratio)

        #Compute optical flow
      #  flow = self.flow_net.compute_flow(prev_crop, next_crop)
        logger.info(f"[FlowTracker] Calling compute_flow() for UUID {self.uuid}")

        flow = self.flow_net.compute_flow(prev_crop, next_crop, debug_uuid=self.uuid,
                                          frame_index=current_frame_index)
        logger.info(f"[FlowTracker] compute_flow() returned for UUID {self.uuid}")

        x, y, w, h = bbox
        rel_x = x - x1
        rel_y = y - y1

        flow_h, flow_w = flow.shape[:2]
        rel_x = max(0, min(rel_x, flow_w - 1))
        rel_y = max(0, min(rel_y, flow_h - 1))
        w = max(1, min(w, flow_w - rel_x))
        h = max(1, min(h, flow_h - rel_y))

        flow_roi = flow[rel_y:rel_y + h, rel_x:rel_x + w]
        flow_x = flow_roi[..., 0]
        flow_y = flow_roi[..., 1]
        magnitude_map = np.sqrt(flow_x ** 2 + flow_y ** 2)

        #Focus on top 30% motion
        threshold = np.percentile(magnitude_map, 70)
        mask = magnitude_map >= threshold

        if np.count_nonzero(mask) < 20:
            logger.info(f"[FlowNet] UUID: {self.uuid} | Insufficient motion — keeping box")
            self.frames_since_last_match += 1
            return bbox

        #Weighted average of dx, dy
        weights = magnitude_map[mask]
        #dx = np.mean(flow_x[mask])
        #dy = np.mean(flow_y[mask])
        dx = np.sum(flow_x[mask] * weights) / np.sum(weights)
        dy = np.sum(flow_y[mask] * weights) / np.sum(weights)
        magnitude = np.sqrt(dx ** 2 + dy ** 2)

        logger.info(f"[FlowNet] UUID: {self.uuid} | dx: {dx:.2f}, dy: {dy:.2f}, mag: {magnitude:.2f}")

        if magnitude < min_motion_threshold:
            self.frames_since_last_match += 1
            return bbox
        else:
            self.frames_since_last_match = 0

        """
        # === Scale adjustment based on motion spread ===
        weights = magnitude_map[mask]
        dx = np.sum(flow_x[mask] * weights) / np.sum(weights)
        dy = np.sum(flow_y[mask] * weights) / np.sum(weights)
        """
        # Apply movement
        x_new = int(x + dx)
        y_new = int(y + dy)

        # === Box Resize based on motion spread ===
        std_x = np.std(flow_x[mask])
        std_y = np.std(flow_y[mask])
        scale_factor = 1.0 + 0.5 * ((std_x + std_y) / 10.0)

        # Clamp scale to never shrink under 90%, never grow beyond 120%
        scale_factor = np.clip(scale_factor, 0.9, 1.2)

        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)

        # Enforce min/max absolute size
        # Clamp to frame
        new_w = max(100, min(new_w, w_frame - x_new))
        new_h = max(200, min(new_h, h_frame - y_new))
        x_new = max(0, min(x_new, w_frame - new_w))
        y_new = max(0, min(y_new, h_frame - new_h))

        #Apply motion and clip to frame bounds
        #x_new = max(0, min(w_frame - w, int(x + dx)))
        #y_new = max(0, min(h_frame - h, int(y + dy)))
        # Save debug crops (optional)
        cv2.imwrite(f"flow_debug_prev_{self.uuid}.jpg", prev_crop)
        cv2.imwrite(f"flow_debug_next_{self.uuid}.jpg", next_crop)
        logger.info(f"[FlowNet] Updated box → x={x_new}, y={y_new}, w={new_w}, h={new_h}, scale={scale_factor:.2f}")
        return x_new, y_new, new_w, new_h

        #return x_new, y_new, w, h