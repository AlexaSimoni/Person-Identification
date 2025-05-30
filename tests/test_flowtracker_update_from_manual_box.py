import os
import sys
import cv2
import numpy as np

# === Make root imports possible ===
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_PATH)

from server.config.config import FLOWNET_MODEL_PATH
from server.FlowNet_Component.FlowNetSWrapper import FlowNetSWrapper
from server.FlowNet_Component.FlowTracker import FlowTracker
from server.Utils.framesGlobals import all_even_frames

# === Input / Output paths ===
img1_path = "test_images/frame1.jpg"
img2_path = "test_images/frame2.jpg"
output_dir = "flow_debug"
uuid = "manual_motion_box"
frame_index_1 = 2
frame_index_2 = 4  # simulate tracking two frames later

# === Load frames ===
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
assert img1 is not None and img2 is not None, "Frame load failed"

h_frame, w_frame = img1.shape[:2]
#manual_box = (10, 10, w_frame // 3, h_frame - 10)
manual_box = (w_frame-(w_frame // 3), 10, w_frame -10, h_frame - 10)
# === Populate globals for FlowTracker ===
all_even_frames.clear()
all_even_frames[frame_index_1] = img1
all_even_frames[frame_index_2] = img2

# === Init tracker ===
flow_net = FlowNetSWrapper(FLOWNET_MODEL_PATH)
tracker = FlowTracker(flow_net=flow_net, uuid=uuid)
tracker.last_box = manual_box
tracker.last_frame_index = frame_index_1
tracker.initial_facenet_box = manual_box  # preserve original size constraints

# === Update tracker to new frame ===
updated_box = tracker.update_track_frame(current_frame_index=frame_index_2)

# === Visualize ===
img1_out = img1.copy()
img2_out = img2.copy()
x1, y1, w1, h1 = manual_box
x2, y2, w2, h2 = updated_box

cv2.rectangle(img1_out, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)  # green = original
cv2.rectangle(img2_out, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)  # red = updated

os.makedirs(output_dir, exist_ok=True)
cv2.imwrite(os.path.join(output_dir, "manual_box_frame1.jpg"), img1_out)
cv2.imwrite(os.path.join(output_dir, "updated_box_frame2.jpg"), img2_out)

print(f"FlowTracker test complete.")
print(f"Original: (x={x1}, y={y1}, w={w1}, h={h1})")
print(f"Updated : (x={x2}, y={y2}, w={w2}, h={h2})")
print(f"Results saved to: {output_dir}")
