import cv2
import numpy as np

#Computes dense optical flow using Farneback's method (CPU-friendly).
#Output: Optical flow field of shape (H, W, 2) where each pixel has (dx, dy).
class SimpleFlowNet:
    def __init__(self, logger=None):
        self.logger = logger

    def compute_flow(self, prev_crop: np.ndarray, next_crop: np.ndarray) -> np.ndarray:
        #Compute dense optical flow between two cropped BGR frames
        #prev_crop: Previous frame (BGR)
        #next_crop: Next frame (BGR)
        #return: Optical flow array (H x W x 2)

        #If the input is BGR, convert to grayscale
        prev_gray = cv2.cvtColor(prev_crop, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_crop, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )

        if self.logger:
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            self.logger.info(f"[SimpleFlowNet] Avg motion magnitude: {np.mean(mag):.2f}")

        return flow