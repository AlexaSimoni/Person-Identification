import logging
import threading
import time

import cv2

from server.FlowNet_Component.TrackingManager import TrackingManager
from server.Utils.framesGlobals import all_even_frames
from server.config.config import USE_FLOWNETS


logger = logging.getLogger(__name__)
logger.info(f"Using {'FlowNetS' if USE_FLOWNETS else 'SimpleFlowNet (Farneback)'} for optical flow.")

#Singleton control to start tracking only once
flow_tracking_started = False
flow_tracking_lock = threading.Lock()

tracking_manager = TrackingManager()

def start_flow_tracking():
    #Start the background FlowNet tracking loop once
    global flow_tracking_started

    with flow_tracking_lock:
        if flow_tracking_started:
            return
        flow_tracking_started = True
        threading.Thread(target=flow_tracking_loop, daemon=True).start()
        logger.info("FlowNet tracking thread started.")

def flow_tracking_loop():
    #Background loop that applies FlowNet tracking to new frames
    #Continuously checks for the latest frame index and updates tracked boxes
    logger.info("FlowNet tracking loop running...")
    last_index = -1

    while True:
        if all_even_frames:
            latest_index = max(all_even_frames.keys())
            if latest_index > last_index:
                last_index = latest_index
                tracking_manager.update_all(latest_index)

        time.sleep(0.03)  #Sleep to avoid busy loop (CPU)

def get_tracking_manager():
    return tracking_manager

def draw_tracking_boxes(frame, frame_index):
    #Draw tracked bounding boxes from FlowNet on the given frame
    #Used during annotation of even frames only
    tracker_manager = get_tracking_manager()

    for tracker in tracker_manager.get_all():
        if tracker.last_box is not None:
            x, y, w, h = tracker.last_box
            label = f"FlowNet | Best: {tracker.best_score:.2f}%"
            logger.info(f"FlowNet box for {tracker.uuid} at frame {frame_index}: {tracker.last_box}")

            #Draw bounding box in orange
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)

            #Label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x = x
            text_y = y - 10 if y - 10 > 10 else y + text_size[1] + 10

            cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5),
                          (text_x + text_size[0], text_y + 5), (0, 165, 255), cv2.FILLED)
            cv2.putText(frame, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
