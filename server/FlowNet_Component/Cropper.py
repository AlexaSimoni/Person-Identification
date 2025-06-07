#import cv2
import numpy as np
from typing import Tuple

class Cropper:
    @staticmethod
    def crop_with_margin(frame: np.ndarray, bbox: Tuple[int, int, int, int], margin_ratio: float = 1.0) -> Tuple[np.ndarray, Tuple[int, int]]:
        #Crop a region around the bounding box with added margin from a full BGR frame
        #Ensures output stays within the image boundaries
        #frame: Full image (BGR)
        #bbox: Bounding box as (x, y, w, h)
        #margin_ratio: Margin to add to width and height (as a fraction)
        #return: (cropped BGR image, top-left corner offset (x1, y1))

        h_frame, w_frame = frame.shape[:2]
        x, y, w, h = bbox

        pad_w = int(w * margin_ratio)
        pad_h = int(h * margin_ratio)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(x + w + pad_w, w_frame)
        y2 = min(y + h + pad_h, h_frame)

        cropped = frame[y1:y2, x1:x2]
        return cropped, (x1, y1)