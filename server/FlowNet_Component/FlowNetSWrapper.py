import os
import sys
import torch
import numpy as np
import cv2
import logging

from torchvision import transforms

from server.Utils.helpers import ensure_dir_exists_if_needed
from server.config.config import ROOT_PATH, FLOWNET_MODEL_PATH
# Add FlowNetPytorch models directory to path
"""
flow_component_path = os.path.join(ROOT_PATH, "server", "FlowNet_Component")
if flow_component_path not in sys.path:
    sys.path.append(flow_component_path)
    """
from server.FlowNet_Component.FlowNetPytorch.models.FlowNetS import FlowNetS
#Wraps the pretrained FlowNetS CNN model for optical flow inference
#from server.config.config import FLOW_VIS_DIR
import server.Utils.framesGlobals as framesGlobals

logger = logging.getLogger(__name__)

class FlowNetSWrapper:
    def __init__(self, checkpoint_path: str):
        print(">>> checkpoint_path =", checkpoint_path)
        print(">>> File exists:", os.path.isfile(checkpoint_path))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FlowNetS(batchNorm=False).to(self.device).eval()

        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        self.transform = transforms.ToTensor()

    def preprocess_frames(self, img1: np.ndarray, img2: np.ndarray) -> torch.Tensor:
        #Convert BGR images to resized RGB tensors and concatenate into FlowNetS input
        #img1: First frame (H, W, 3) BGR
        #img2: Second frame (H, W, 3) BGR
        #return: Input tensor of shape (1, 6, H, W)

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Resize to 512x384 (FlowNetS default input)
        img1 = cv2.resize(img1, (512, 384))
        img2 = cv2.resize(img2, (512, 384))

        tensor1 = self.transform(img1)
        tensor2 = self.transform(img2)

        input_tensor = torch.cat([tensor1, tensor2], dim=0).unsqueeze(0).to(self.device)
        return input_tensor

    def compute_flow(self, img1: np.ndarray, img2: np.ndarray, debug_uuid: str = None, frame_index: int = None) -> np.ndarray:
        #Compute optical flow between two frames using FlowNetS.
        #img1: First frame (BGR)
        #img2: Second frame (BGR)
        #return: Flow as (H, W, 2)
        logger.info(f"[FlowNetSWrapper] compute_flow() called | uuid={debug_uuid} frame={frame_index}")

        input_tensor = self.preprocess_frames(img1, img2)

        with torch.no_grad():
            flow = self.model(input_tensor)[0].cpu().numpy()

        #Transpose from (2, H, W) to (H, W, 2)
        flow = np.transpose(flow, (1, 2, 0))

        # === Debug: Print average flow magnitude ===
        flow_magnitude = np.linalg.norm(flow, axis=2)
        avg_magnitude = np.mean(flow_magnitude)
        logger.info(f"[FlowNetSWrapper] Avg flow magnitude: {avg_magnitude:.3f}")

        # === Optional: Save visual flow map ===
        try:
            if debug_uuid is not None and frame_index is not None and framesGlobals.dir_path is not None:
                logger.info(f"[FlowNetSWrapper] Attempting to save flow vis to: {framesGlobals.dir_path}")
                #os.makedirs(dir_path, exist_ok=True)
                save_path = os.path.join(framesGlobals.dir_path, f"{debug_uuid}_{frame_index}.jpg")
                if os.path.isfile(save_path):
                    logger.info(f"[FlowNetSWrapper] Successfully saved debug image at: {save_path}")
                else:
                    logger.error(f"[FlowNetSWrapper] Failed to save debug image — file not found: {save_path}")

                vis = self.visualize_flow(flow)
                cv2.imwrite(save_path, vis)
                logger.info(f"[FlowNetSWrapper] Successfully wrote to: {save_path}")
        except Exception as e:
            logger.error(f"[FlowNetSWrapper] Failed to save flow visualization: {e}")

        return flow

    def visualize_flow(self, flow: np.ndarray) -> np.ndarray:
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def get_debug_dir_from_video(self, video_path: str) -> str:
        video_dir = os.path.dirname(video_path)
        debug_dir = os.path.join(video_dir, "flow_vis")
        return debug_dir
