"""
Pothole Detection Model
Uses EfficientNet with transfer learning
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
import os
import logging

from utils.config import Config

logger = logging.getLogger(__name__)

class PotholeDetector:
    """Pothole detection using EfficientNet"""

    def __init__(self):
        self.config = Config()
        self.device = torch.device(self.config.device)
        self.model = None
        self.transform = None
        self._load_model()

    def _load_model(self):
        """Load EfficientNet model"""
        try:
            # Load pretrained EfficientNet
            self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
            self.model.to(self.device)
            self.model.eval()

            # Image transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            # Try to load fine-tuned weights if available
            if os.path.exists(self.config.pothole_model_path):
                self.model.load_state_dict(torch.load(self.config.pothole_model_path, map_location=self.device))
                logger.info("Fine-tuned pothole model loaded")
            else:
                logger.info("Using pretrained EfficientNet (not fine-tuned for potholes)")

        except Exception as e:
            logger.error(f"Failed to load pothole model: {e}")
            self.model = None

    def detect(self, image: np.ndarray) -> list:
        """Detect potholes in image"""
        if self.model is None:
            return []

        try:
            # Sliding window approach for detection
            potholes = []
            window_size = 128
            stride = 64

            height, width = image.shape[:2]

            for y in range(0, height - window_size, stride):
                for x in range(0, width - window_size, stride):
                    # Extract window
                    window = image[y:y+window_size, x:x+window_size]

                    # Classify window
                    if self._classify_window(window):
                        potholes.append({
                            'bbox': [x, y, x+window_size, y+window_size],
                            'confidence': 0.8  # Placeholder
                        })

            return potholes

        except Exception as e:
            logger.error(f"Pothole detection failed: {e}")
            return []

    def _classify_window(self, window: np.ndarray) -> bool:
        """Classify if window contains pothole"""
        try:
            # Convert to PIL
            pil_window = Image.fromarray(cv2.cvtColor(window, cv2.COLOR_BGR2RGB))

            # Transform
            tensor = self.transform(pil_window).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = torch.softmax(outputs, dim=1)
                pothole_prob = probabilities[0][1].item()

            return pothole_prob > 0.5

        except Exception as e:
            logger.error(f"Window classification failed: {e}")
            return False