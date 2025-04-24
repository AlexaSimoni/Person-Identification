import cv2
import numpy as np

# ---------------------------------------------------
# Class: SimpleFlowNet
# ---------------------------------------------------
# This class handles computing dense optical flow using Farneback's method.
# INPUT: two RGB frames
# OUTPUT: a dense optical flow map with shape (H, W, 2), where each pixel
#         contains the motion vector (dx, dy) from prev_frame to next_frame.
class SimpleFlowNet:
    def __init__(self):
        self.prev_gray = None  # store previous frame in grayscale

    def compute_flow(self, prev_frame, next_frame):
        # Step 1: Convert frames to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        # Step 2: Compute dense optical flow using the Farneback method
        #         which estimates motion for every pixel
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray,
            None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )

        # OUTPUT: flow[y, x] = (dx, dy)
        return flow
