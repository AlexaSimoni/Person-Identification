import numpy as np
from server.FlowNet_Component.SimpleFlowNet import SimpleFlowNet

# ==========================================
# FlowNet_Component: Optical Flow Tracking
# ==========================================
# This module tracks a previously detected object (e.g. a person)
# using optical flow to update the bounding box over time.
# It avoids rerunning YOLO on every frame by estimating motion.

# Adjustable frame interval for when to apply tracking updates
frame_update_interval = 2  # Only update every N frames (for performance)

# ---------------------------------------------------
# Class: FlowTracker
# ---------------------------------------------------
# This class manages the tracking of a previously-detected region (box)
# using the optical flow field from SimpleFlowNet.
#
# Usage:
# - After detecting a person with YOLO and verifying with FaceNet,
#   pass the bounding box to this tracker.
# - It will update the box based on motion in later frames.
#
# INPUTS:
#   current_frame: RGB frame at current timestep
#   current_box: (x, y, w, h) of the last known person location
#
# OUTPUT:
#   Updated box: (x + dx, y + dy, w, h), shifted by motion
class FlowTracker:
    def __init__(self):
        self.flow_net = SimpleFlowNet()  # optical flow computer
        self.last_frame = None           # previously processed frame
        self.last_box = None             # previously tracked bounding box
        self.frame_count = 0             # count of total frames seen
        self.frame_update_interval = frame_update_interval  # Track every N frames only (adjustable)

    def update_track_frame(self, current_frame, current_box):
        # Step 0: Frame skipping — only apply update every N frames
        self.frame_count += 1
        if self.frame_count % frame_update_interval != 0:
            return current_box  # No update — return previous box

        # Step 1: If this is the first frame, just store and return the input box
        if self.last_frame is None or self.last_box is None:
            self.last_frame = current_frame
            self.last_box = current_box
            return current_box

        # Step 2: Compute optical flow from last_frame → current_frame
        flow = self.flow_net.compute_flow(self.last_frame, current_frame)

        # Step 3: Use the flow in the region of the old box to estimate motion
        x, y, w, h = self.last_box
        h_frame, w_frame, _ = current_frame.shape

        # Clamp box to image dimensions to avoid overflow
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = max(1, min(w, w_frame - x))
        h = max(1, min(h, h_frame - y))

        # Slice the flow map to the region of the previous box
        flow_region = flow[y:y + h, x:x + w]  # (h, w, 2) - Region of interest for motion vectors
        dx = int(np.mean(flow_region[..., 0]))  # average flow vector horizontal motion
        dy = int(np.mean(flow_region[..., 1]))  # average flow vector vertical motion

        # Step 4: Shift box according to estimated motion
        new_x = max(0, min(x + dx, w_frame - w))
        new_y = max(0, min(y + dy, h_frame - h))
        new_box = (new_x, new_y, w, h)

        # Step 5: Update internal state
        self.last_box = new_box
        self.last_frame = current_frame

        # OUTPUT: updated bounding box
        return new_box
