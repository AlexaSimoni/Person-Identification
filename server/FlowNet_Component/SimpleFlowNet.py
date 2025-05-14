import cv2
import numpy as np
#from server.Yolo_Componenet.Frame import Frame
#from server.Yolo_Componenet.Yolo_Utils import all_even_frames



# ---------------------------------------------------
# Class: SimpleFlowNet
# ---------------------------------------------------
# This class handles computing dense optical flow using Farneback's method.
# INPUT: two RGB frames
# OUTPUT: a dense optical flow map with shape (H, W, 2), where each pixel
#         contains the motion vector (dx, dy) from prev_frame to next_frame.
class SimpleFlowNet:
    def __init__(self, logger):
        self.prev_gray = None  # store previous frame in grayscale
        self.logger = logger

   # def compute_flow(self, prev_frame, next_frame):
    """
    def compute_flow(self, frame_index_1, frame_index_2):
        frame_obj1 = Frame(frame_index_1)
        frame_obj2 = Frame(frame_index_2)
        # Step 1: Convert frames to grayscale
        #prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(frame_obj1, cv2.COLOR_BGR2GRAY)
        #next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(frame_obj2, cv2.COLOR_BGR2GRAY)

        # Step 2: Compute dense optical flow using the Fa method
        # which estimates motion for every pixel between two frames,
        # result is a flow field: an array of (dx, dy) vectors
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )
        # OUTPUT: flow[y, x] = (dx, dy)
        return flow
"""
    def compute_flow(self, prev_crop, next_crop) -> np.ndarray:
        """
         Compute dense optical flow between two cropped regions.

         :param prev_crop: Cropped region from previous frame (expected BGR)
         :param next_crop: Cropped region from next/current frame (expected BGR)
         :return: Optical flow field (H x W x 2)
         """
        # If the input is BGR, convert to grayscale
        prev_gray = cv2.cvtColor(prev_crop, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_crop, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )
        return flow