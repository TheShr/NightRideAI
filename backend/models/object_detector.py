"""
Object Detection Model
Uses YOLOv8 for object detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
import logging

from utils.config import Config

logger = logging.getLogger(__name__)

class ObjectDetector:
    """Object detection using YOLOv8"""

    def __init__(self):
        self.config = Config()
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load YOLO model"""
        try:
            self.model = YOLO(self.config.yolo_model)
            logger.info("YOLOv8 model loaded")

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None

    def detect(self, image: np.ndarray) -> list:
        """Detect objects in image"""
        if self.model is None:
            return []

        try:
            # Run inference
            results = self.model(image, conf=0.5, classes=[0, 1, 2, 16, 17])  # person, bicycle, car, dog, horse

            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())

                    class_names = {
                        0: 'person',
                        1: 'bicycle',
                        2: 'car',
                        16: 'dog',
                        17: 'horse'
                    }

                    detections.append({
                        'class': class_names.get(cls, 'unknown'),
                        'confidence': float(conf),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })

            return detections

        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []