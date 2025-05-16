import numpy as np
import logging
from server.Utils.framesGlobals import all_even_frames, detections_frames
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
        if self.last_frame_index not in all_even_frames or current_frame_index not in all_even_frames:
            return self.last_box

        prev_frame = all_even_frames[self.last_frame_index]
        curr_frame = all_even_frames[current_frame_index]

        #Update box using optical flow
        updated_box = self.update_bbox_with_optical_flow(prev_frame, curr_frame, self.last_box)

        self.last_box = updated_box
        self.last_frame_index = current_frame_index

        return self.last_box

    def update_bbox_with_optical_flow(self, prev_frame: np.ndarray, next_frame: np.ndarray, bbox: tuple,
                                      min_motion_threshold: float = 1.0) -> tuple:
        #Use optical flow to update the bounding box between two frames
        h_frame, w_frame = prev_frame.shape[:2]

        #Crop both frames with margin
        prev_crop, (x1, y1) = Cropper.crop_with_margin(prev_frame, bbox, margin_ratio=0.25)
        next_crop, _ = Cropper.crop_with_margin(next_frame, bbox, margin_ratio=0.25)

        #Compute optical flow
        flow = self.flow_net.compute_flow(prev_crop, next_crop)

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

        # Focus on top 30% motion
        threshold = np.percentile(magnitude_map, 70)
        mask = magnitude_map >= threshold

        if np.count_nonzero(mask) < 10:
            logger.info(f"[FlowNet] UUID: {self.uuid} | Insufficient motion — keeping box")
            return bbox

        dx = np.mean(flow_x[mask])
        dy = np.mean(flow_y[mask])
        magnitude = np.sqrt(dx ** 2 + dy ** 2)

        logger.info(f"[FlowNet] UUID: {self.uuid} | dx: {dx:.2f}, dy: {dy:.2f}, mag: {magnitude:.2f}")

        if magnitude < min_motion_threshold:
            return bbox

        #Apply motion and clip to frame bounds
        x_new = max(0, min(w_frame - w, int(x + dx)))
        y_new = max(0, min(h_frame - h, int(y + dy)))

        return x_new, y_new, w, h