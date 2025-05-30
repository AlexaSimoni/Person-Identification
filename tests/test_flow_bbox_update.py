import os
import cv2
import sys

import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.FlowNet_Component.FlowNetSWrapper import FlowNetSWrapper
from server.FlowNet_Component.Cropper import Cropper
from server.config.config import FLOWNET_MODEL_PATH

# === Paths ===
img1_path = "../test_images/frame1.jpg"
img2_path = "../test_images/frame2.jpg"
output_dir = "../flow_debug"
uuid = "bbox_test"
frame_index = 1

# === Load frames ===
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
assert img1 is not None and img2 is not None, "Failed to load input images."

h_frame, w_frame = img1.shape[:2]

# === Define bbox: left vertical third ===
bbox = (4, 4, w_frame // 3, h_frame)

# === Init FlowNetS ===
net = FlowNetSWrapper(checkpoint_path=FLOWNET_MODEL_PATH)

# === Crop + compute flow ===
crop1, (x1, y1) = Cropper.crop_with_margin(img1, bbox, margin_ratio=1.0)
crop2, _ = Cropper.crop_with_margin(img2, bbox, margin_ratio=1.0)
flow = net.compute_flow(crop1, crop2, debug_uuid=uuid, frame_index=frame_index)

# === Apply flow to update box ===
x, y, w, h = bbox
rel_x = x - x1
rel_y = y - y1
flow_roi = flow[rel_y:rel_y + h, rel_x:rel_x + w]

flow_x = flow_roi[..., 0]
flow_y = flow_roi[..., 1]
mag = np.sqrt(flow_x ** 2 + flow_y ** 2)
threshold = np.percentile(mag, 50)
mask = mag >= threshold

weights = mag[mask]
dx = np.sum(flow_x[mask] * weights) / np.sum(weights)
dy = np.sum(flow_y[mask] * weights) / np.sum(weights)

x_new = int(x + dx)
y_new = int(y + dy)

# === Resize box (motion spread) ===
std_x = np.std(flow_x[mask])
std_y = np.std(flow_y[mask])
scale = np.clip(1.0 + 1.5 * ((std_x + std_y) / 10.0), 0.9, 1.5)
new_w = int(np.clip(w * scale, 50, w_frame - x_new))
new_h = int(np.clip(h * scale, 100, h_frame - y_new))
x_new = max(0, min(w_frame - new_w, x_new))
y_new = max(0, min(h_frame - new_h, y_new))

updated_bbox = (x_new, y_new, new_w, new_h)

# === Draw boxes ===
img1_boxed = img1.copy()
img2_boxed = img2.copy()
cv2.rectangle(img1_boxed, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green: original
cv2.rectangle(img2_boxed, (x_new, y_new), (x_new + new_w, y_new + new_h), (0, 0, 255), 2)  # Red: updated

# === Save results ===
os.makedirs(output_dir, exist_ok=True)
cv2.imwrite(os.path.join(output_dir, "bbox_original_frame1.jpg"), img1_boxed)
cv2.imwrite(os.path.join(output_dir, "bbox_updated_frame2.jpg"), img2_boxed)

print(f"FlowNet bbox test done. Output saved in {output_dir}")
print(f"Original bbox: {bbox}")
print(f"Updated bbox : {updated_bbox}")
