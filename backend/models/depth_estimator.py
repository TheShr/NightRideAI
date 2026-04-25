"""
Monocular Depth Estimation Model
Uses MiDaS DPT Large for depth estimation
"""

import cv2
import numpy as np
import torch
from PIL import Image
import logging

from utils.config import Config

logger = logging.getLogger(__name__)

class DepthEstimator:
    """Depth estimation using MiDaS"""

    def __init__(self):
        self.config = Config()
        self.device = torch.device(self.config.device)
        self.model = None
        self.transform = None
        self._load_model()

    def _load_model(self):
        """Load MiDaS model"""
        try:
            # Use torch hub for MiDaS
            self.model = torch.hub.load('intel-isl/MiDaS', 'DPT_Large', pretrained=True)
            self.model.to(self.device)
            self.model.eval()

            # Load transforms
            midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            self.transform = midas_transforms.dpt_transform

            logger.info("MiDaS DPT Large model loaded")

        except Exception as e:
            logger.error(f"Failed to load MiDaS model: {e}")
            self.model = None

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth from image"""
        if self.model is None:
            # Fallback: return dummy depth map
            return np.ones((image.shape[0], image.shape[1]), dtype=np.float32) * 10.0

        try:
            # Convert to RGB NumPy array for MiDaS transform
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_image = np.asarray(rgb_image)

            # Transform
            input_batch = self.transform(input_image).to(self.device)

            with torch.no_grad():
                prediction = self.model(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image.shape[:2],
                    mode='bicubic',
                    align_corners=False
                ).squeeze()

            depth_map = prediction.cpu().numpy()
            return depth_map

        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return np.ones((image.shape[0], image.shape[1]), dtype=np.float32) * 10.0