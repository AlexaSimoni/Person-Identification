import numpy as np
import logging
import threading
from server.Utils.framesGlobals import all_even_frames, detections_frames
from server.FlowNet_Component.Cropper import Cropper
from server.FlowNet_Component.SimpleFlowNet import SimpleFlowNet
from server.config.config import SIMILARITY_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlowTracker:
    def __init__(self, flow_net, uuid):
        #self.flow_net = SimpleFlowNet()  # optical flow computer
        self.flow_net = flow_net
        self.uuid = uuid
        self.last_frame_index = None           # previously processed frame
        self.last_box = None             # previously tracked bounding box
        self.best_score = 0.0

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

        # Update box with optical flow
        updated_box = self.update_bbox_with_optical_flow(self.flow_net, prev_frame, curr_frame, self.last_box)

        # If FaceNet detection is available for current frame, compare box overlap
        if current_frame_index in detections_frames:
            coords, similarity = detections_frames[current_frame_index]
            x1, y1, x2, y2 = coords
            facenet_box = (x1, y1, x2 - x1, y2 - y1)

            iou = self.compute_iou(facenet_box, updated_box)
            logger.info(f"[FlowNet-IoU] Frame {current_frame_index} IoU with FaceNet: {iou:.2f}")

            if iou >= 0.3:
                logger.info(f"[FlowNet-IoU] Accepting updated box at frame {current_frame_index}")
                self.last_box = updated_box
            else:
                logger.warning(f"[FlowNet-IoU] IoU too low ({iou:.2f}), keeping previous box")
                # Do not update the box if it likely drifted off
        else:
            # No FaceNet box to compare, update anyway
            self.last_box = updated_box

        """
        # --- FlowNet override with FaceNet detection if available ---
        facenet_box = None
        if current_frame_index in detections_frames:
            coords, similarity = detections_frames[current_frame_index]
            if similarity >= SIMILARITY_THRESHOLD:
                x1, y1, x2, y2 = coords
                facenet_box = (x1, y1, x2 - x1, y2 - y1)  # Convert to (x, y, w, h)
                logger.info(f"[FlowNet] Using FaceNet box override for {self.uuid} at frame {current_frame_index}")

        # If FaceNet box available, use it instead of previous FlowNet box
        base_box = facenet_box if facenet_box is not None else self.last_box
        if base_box is None:
            self.last_frame_index = current_frame_index
            return None  # Nothing to track yet

        # --- Optical flow update ---
        updated_box = self.update_bbox_with_optical_flow(self.flow_net, prev_frame, curr_frame, base_box)

        """
        #self.last_box = updated_box
        self.last_frame_index = current_frame_index
        #return updated_box
        return self.last_box

    """
    def update_track_frame(self, current_frame_index):
       
        if self.last_box is None or self.last_frame_index is None:
            self.last_frame_index = current_frame_index
            return self.last_box


        if self.last_frame_index not in all_even_frames or current_frame_index not in all_even_frames:
            return self.last_box
       
        prev_frame = all_even_frames[self.last_frame_index]
        curr_frame = all_even_frames[current_frame_index]

        updated_box = self.update_bbox_with_optical_flow(self.flow_net, prev_frame, curr_frame, self.last_box)

        self.last_box = updated_box
        self.last_frame_index = current_frame_index
        return updated_box
    """


    def update_bbox_with_optical_flow(self,flow_net: SimpleFlowNet, prev_frame: np.ndarray, next_frame: np.ndarray,
                                      bbox: tuple, min_motion_threshold: float = 1.0) -> tuple:
        """
        Update bounding box using optical flow computed in a small ROI via Cropper and SimpleFlowNet.

        :param flow_net: An instance of SimpleFlowNet
        :param prev_frame: Previous full BGR frame
        :param next_frame: Next full BGR frame
        :param bbox: Tuple (x, y, w, h) representing the original bounding box
        :param min_motion_threshold: Minimum flow magnitude to consider motion valid
        :return: Tuple (x_new, y_new, w, h) as the updated bounding box
        """
        # Step 1: Crop both frames using Cropper
        prev_crop, (x1, y1) = Cropper.crop_with_margin(prev_frame, bbox)
        next_crop, _ = Cropper.crop_with_margin(next_frame, bbox)

        # Step 2: Compute optical flow
        flow = flow_net.compute_flow(prev_crop, next_crop)

        # Step 3: Extract flow region inside original box area
        x, y, w, h = bbox
        rel_x = x - x1
        rel_y = y - y1

        flow_h, flow_w = flow.shape[:2]
        rel_x = max(0, min(rel_x, flow_w - 1))
        rel_y = max(0, min(rel_y, flow_h - 1))
        w = max(1, min(w, flow_w - rel_x))
        h = max(1, min(h, flow_h - rel_y))

        flow_roi = flow[rel_y:rel_y + h, rel_x:rel_x + w]

        # Step 4: Compute flow magnitude
        dx = np.mean(flow_roi[..., 0])
        dy = np.mean(flow_roi[..., 1])
        magnitude = np.sqrt(dx ** 2 + dy ** 2)

        # Log the flow vector
        logger.info(f"[FlowNet] UUID: {self.uuid} | dx: {dx:.2f}, dy: {dy:.2f}, magnitude: {magnitude:.2f}")
       # Step 5: Check motion threshold
        if magnitude < min_motion_threshold:
            return bbox  # Not enough motion — keep original box

        # Step 6: Apply flow to update box
        x_new = max(0, x + int(dx))
        y_new = max(0, y + int(dy))

        return x_new, y_new, w, h

