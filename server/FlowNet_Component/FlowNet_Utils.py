import numpy as np
import logging
import threading
#import asyncio

import time
#from server.FlowNet_Component.SimpleFlowNet import SimpleFlowNet
#from server.Yolo_Componenet.Yolo_Utils import all_even_frames
#from server.FlowNet_Component.Cropper import Cropper
#from server.FlowNet_Component.SimpleFlowNet import SimpleFlowNet
#from server.Yolo_Componenet.Yolo_Utils import all_even_frames
#from server.FlowNet_Component.FlowNet_Utils import update_bbox_with_optical_flow
from server.FlowNet_Component.TrackingManager import TrackingManager
#from server.FlowNet_Component.TrackingManager import tracker_manager
#from server.Yolo_Componenet.Yolo_Utils import all_even_frames
from server.Utils.framesGlobals import annotated_frames, detections_frames, all_even_frames


# ==========================================
# FlowNet_Component: Optical Flow Tracking
# ==========================================
# This module tracks a previously detected object (e.g. a person)
# using optical flow to update the bounding box over time.
# It avoids rerunning YOLO on every frame by estimating motion.

# Adjustable frame interval for when to apply tracking updates
#frame_update_interval = 2  # Only update every N frames (for performance)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

flow_tracking_started = False
flow_tracking_lock = threading.Lock()
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


def start_flow_tracking():
    global flow_tracking_started
    with flow_tracking_lock:
        if flow_tracking_started:
            return
        flow_tracking_started = True
        threading.Thread(target=flow_tracking_loop, daemon=True).start()
        logger.info("FlowNet tracking thread started.")

def flow_tracking_loop():
    logger.info("FlowNet tracking loop running...")
    last_index = -1

    while True:
        if all_even_frames:
            latest_index = max(all_even_frames.keys())
            if latest_index > last_index:
                last_index = latest_index
                TrackingManager.update_all(latest_index)

        time.sleep(0.03)  # sleep to avoid busy loop
