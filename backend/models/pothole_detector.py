"""
Pothole Detection Model
Uses EfficientNet with transfer learning
"""

import cv2
import numpy as np
import os
import logging
from PIL import Image

from utils.config import Config

logger = logging.getLogger(__name__)

class PotholeDetector:
    """Pothole detection using EfficientNet"""

    def __init__(self):
        self.config = Config()
        self.torch = None
        self.timm = None
        self.transforms = None
        self.device = None
        self.model = None
        self.model_loaded = False
        self.transform = None
        self._load_model()

    def _load_model(self):
        """Load EfficientNet model"""
        try:
            import torch
            import torchvision.transforms as transforms
            import timm

            self.torch = torch
            self.timm = timm
            self.transforms = transforms
            self.device = self.torch.device(self.config.device)

            # Load pretrained EfficientNet
            self.model = self.timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
            self.model.to(self.device)
            self.model.eval()

            # Image transforms
            self.transform = self.transforms.Compose([
                self.transforms.Resize((224, 224)),
                self.transforms.ToTensor(),
                self.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            # Try to load fine-tuned weights if available
            if os.path.exists(self.config.pothole_model_path):
                self.model.load_state_dict(self.torch.load(self.config.pothole_model_path, map_location=self.device))
                self.model_loaded = True
                logger.info("Fine-tuned pothole model loaded")
            else:
                self.model = None
                self.model_loaded = False
                logger.warning(
                    f"Pothole model weights not found at {self.config.pothole_model_path}. "
                    "Pothole detection will be disabled until a trained model is provided."
                )

        except Exception as e:
            logger.error(f"Failed to load pothole model: {e}")
            self.model = None
            self.model_loaded = False

    def detect(self, image: np.ndarray) -> list:
        """Detect potholes in image"""
        if not self.model_loaded or self.model is None:
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
                    confidence = self._classify_window(window)
                    if confidence is not None and confidence > 0.5:
                        potholes.append({
                            'bbox': [x, y, x+window_size, y+window_size],
                            'confidence': float(confidence)
                        })

            return potholes

        except Exception as e:
            logger.error(f"Pothole detection failed: {e}")
            return []

    def _classify_window(self, window: np.ndarray) -> float:
        """Classify if window contains pothole and return probability"""
        try:
            # Convert to PIL
            pil_window = Image.fromarray(cv2.cvtColor(window, cv2.COLOR_BGR2RGB))

            # Transform
            tensor = self.transform(pil_window).unsqueeze(0).to(self.device)

            with self.torch.no_grad():
                outputs = self.model(tensor)
                probabilities = self.torch.softmax(outputs, dim=1)
                pothole_prob = probabilities[0][1].item()

            return float(pothole_prob)

        except Exception as e:
            logger.error(f"Window classification failed: {e}")
            return None