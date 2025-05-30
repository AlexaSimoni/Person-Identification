import os
import cv2
import sys

import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.FlowNet_Component.FlowNetSWrapper import FlowNetSWrapper
from server.config.config import FLOWNET_MODEL_PATH

# === Input paths ===
img1_path = "test_images/frame1.jpg"
img2_path = "test_images/frame2.jpg"
output_dir = "flow_debug"
uuid = "motion_box"
frame_index = 2

# === Load images ===
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
assert img1 is not None and img2 is not None, "Failed to load frames"

# === Compute flow ===
net = FlowNetSWrapper(checkpoint_path=FLOWNET_MODEL_PATH)
flow = net.compute_flow(img1, img2, debug_uuid=uuid, frame_index=frame_index)

# === Calculate motion magnitude ===
flow_x = flow[..., 0]
flow_y = flow[..., 1]
magnitude = np.sqrt(flow_x ** 2 + flow_y ** 2)

# === Threshold top 5% motion pixels ===
motion_threshold = np.percentile(magnitude, 95)
motion_mask = magnitude > motion_threshold

# === Get coordinates of motion pixels ===
ys, xs = np.where(motion_mask)
if len(xs) == 0 or len(ys) == 0:
    print("No significant motion detected")
    exit()

# === Bounding box from motion coordinates ===
x_min, x_max = xs.min(), xs.max()
y_min, y_max = ys.min(), ys.max()
bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

# === Draw proposed box ===
output_img = img2.copy()
cv2.rectangle(output_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blue box

# === Save result ===
os.makedirs(output_dir, exist_ok=True)
cv2.imwrite(os.path.join(output_dir, "motion_based_box_frame2.jpg"), output_img)

print(f"New motion-based box created: {bbox}")
print(f"Saved result in {output_dir}")
