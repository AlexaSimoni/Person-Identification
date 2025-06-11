import os
import torch
import numpy as np
import cv2
import logging

from torchvision import transforms
from server.FlowNet_Component.FlowNetPytorch.models.FlowNetS import FlowNetS
import server.Utils.framesGlobals as framesGlobals

logger = logging.getLogger(__name__)

# FlowNetSWrapper class wraps the pretrained FlowNetS neural network model for estimating optical flow between image frames
class FlowNetSWrapper:
    # Inputs: checkpoint_path (Path to the pretrained model checkpoint)
    # Output: Flow tensors representing pixel-wise motion vectors between two consecutive frames
    def __init__(self, checkpoint_path: str):
        #print(">>> checkpoint_path =", checkpoint_path)
        #print(">>> File exists:", os.path.isfile(checkpoint_path))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FlowNetS(batchNorm=False).to(self.device).eval()
        # Load pretrained weights
        if os.path.isfile(checkpoint_path):
            logger.info(f"[FlowNetSWrapper] checkpoint_path = {checkpoint_path} | File exists: = {os.path.isfile(checkpoint_path)}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        # Define image transform to tensor
        self.transform = transforms.ToTensor()

    # Convert BGR images to resized RGB tensors and concatenate into FlowNetS input
    # Inputs: img1, img2 (frames BGR images (H, W, 3))
    # Output: 4D tensor of shape (1, 6, H, W)
    def preprocess_frames(self, img1: np.ndarray, img2: np.ndarray) -> torch.Tensor:
        # Convert OpenCV BGR to RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        # Resize to 512x384 (FlowNetS default input)
        img1 = cv2.resize(img1, (512, 384))
        img2 = cv2.resize(img2, (512, 384))
        # Convert images to torch tensors
        tensor1 = self.transform(img1)
        tensor2 = self.transform(img2)
        # Concatenate along channel dimension -> shape: (6, H, W)
        input_tensor = torch.cat([tensor1, tensor2], dim=0).unsqueeze(0).to(self.device)
        return input_tensor

    # Compute dense optical flow between two frames using FlowNetS
    # Inputs: img1, img2 (frames BGR images (H, W, 3)), debug_uuid (debug flow id), frame_index
    # Output: optical flow as (H, W, 2) representing x, y motion vectors
    def compute_flow(self, img1: np.ndarray, img2: np.ndarray, debug_uuid: str = None, frame_index: int = None) -> np.ndarray:
        logger.info(f"[FlowNetSWrapper] compute_flow() called | uuid={debug_uuid} frame={frame_index}")
        input_tensor = self.preprocess_frames(img1, img2)   # Prepare input tensor for model inference
        with torch.no_grad():   # Run the model without gradient tracking
            flow = self.model(input_tensor)[0].cpu().numpy()

        flow = np.transpose(flow, (1, 2, 0))    # (2, H, W) to (H, W, 2)
        # Print average flow magnitude
        flow_magnitude = np.linalg.norm(flow, axis=2)
        avg_magnitude = np.mean(flow_magnitude)
        logger.info(f"[FlowNetSWrapper] Avg flow magnitude: {avg_magnitude:.3f}")

        # Optional: Save visual flow map images
        try:
            if debug_uuid is not None and frame_index is not None and framesGlobals.dir_path is not None:
                logger.info(f"[FlowNetSWrapper] Attempting to save flow vis to: {framesGlobals.dir_path}")
                save_path = os.path.join(framesGlobals.dir_path, f"{debug_uuid}_{frame_index}.jpg")
                vis = self.visualize_flow(flow)
                cv2.imwrite(save_path, vis)
                logger.info(f"[FlowNetSWrapper] Flow visualization saved at: {save_path}")
        except Exception as e:
            logger.error(f"[FlowNetSWrapper] Failed to save flow visualization: {e}")

        return flow

    # Converts flow vectors into a color-coded image (HSV -> BGR) for visualization
    # Input: flow ((H, W, 2) of flow vectors)
    # Output: BGR image visualizing direction and magnitude of motion
    def visualize_flow(self, flow: np.ndarray) -> np.ndarray:
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)   # Initialize HSV image with zeros
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  # Convert Cartesian flow to polar coordinates (magnitude and angle)
        hsv[..., 0] = ang * 180 / np.pi / 2  # Direction encoded as color
        hsv[..., 1] = 255  # Full saturation
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Magnitude as brightness
        # Convert HSV image to BGR for displaying
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)