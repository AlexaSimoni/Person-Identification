from server.FlowNet_Component.FlowNetSWrapper import FlowNetSWrapper
from server.config.config import FLOWNET_MODEL_PATH
import server.Utils.framesGlobals as framesGlobals
import os
import cv2
import sys

import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_flownets_wrapper():
    TEST_IMAGE_1 = "test_images/frame1.jpg"
    TEST_IMAGE_2 = "test_images/frame2.jpg"
    DEBUG_UUID = "test_uuid"
    FRAME_INDEX = 1

    assert os.path.isfile(FLOWNET_MODEL_PATH), f"Model not found at: {FLOWNET_MODEL_PATH}"
    assert os.path.isfile(TEST_IMAGE_1), f"Test image not found: {TEST_IMAGE_1}"
    assert os.path.isfile(TEST_IMAGE_2), f"Test image not found: {TEST_IMAGE_2}"

    # ✅ Set debug output directory
    framesGlobals.dir_path = "../flow_debug"
    os.makedirs(framesGlobals.dir_path, exist_ok=True)

    img1 = cv2.imread(TEST_IMAGE_1)
    img2 = cv2.imread(TEST_IMAGE_2)
    assert img1 is not None and img2 is not None, "Could not read one of the test images."

    flownets = FlowNetSWrapper(checkpoint_path=FLOWNET_MODEL_PATH)
    flow = flownets.compute_flow(img1, img2, debug_uuid=DEBUG_UUID, frame_index=FRAME_INDEX)

    assert flow.shape[2] == 2, f"Flow output should have 2 channels (dx, dy), got shape: {flow.shape}"
    magnitude = np.linalg.norm(flow, axis=2)
    avg_magnitude = np.mean(magnitude)
    print(f"Flow computed successfully. Avg magnitude = {avg_magnitude:.3f}")

if __name__ == "__main__":
    test_flownets_wrapper()
