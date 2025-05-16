import os
import sys
import torch
import numpy as np
import cv2
from torchvision import transforms
from server.config.config import ROOT_PATH, FLOWNET_MODEL_PATH
# Add FlowNetPytorch models directory to path
"""
flow_component_path = os.path.join(ROOT_PATH, "server", "FlowNet_Component")
if flow_component_path not in sys.path:
    sys.path.append(flow_component_path)
    """
from server.FlowNet_Component.FlowNetPytorch.models.FlowNetS  import FlowNetS
#Wraps the pretrained FlowNetS CNN model for optical flow inference
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

    def compute_flow(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        #Compute optical flow between two frames using FlowNetS.
        #img1: First frame (BGR)
        #img2: Second frame (BGR)
        #return: Flow as (H, W, 2)

        input_tensor = self.preprocess_frames(img1, img2)

        with torch.no_grad():
            flow = self.model(input_tensor)[0].cpu().numpy()

        #Transpose from (2, H, W) to (H, W, 2)
        flow = np.transpose(flow, (1, 2, 0))
        return flow