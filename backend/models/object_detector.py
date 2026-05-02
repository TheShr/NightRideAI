"""
Object Detection Model
Uses YOLOv8 for object detection
"""

import cv2
import numpy as np
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
            from ultralytics import YOLO
            self.model = YOLO(self.config.yolo_model)
            logger.info(f"YOLOv8 model loaded from {self.config.yolo_model}")

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None

    def detect(self, image: np.ndarray) -> list:
        """Detect objects in image"""
        if self.model is None:
            return []

        try:
            # Run inference on all available classes
            results = self.model(image, conf=0.35)

            detections = []
            for result in results:
                names = result.names if hasattr(result, 'names') else {}
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = names.get(cls, str(cls))

                    detections.append({
                        'class': class_name,
                        'confidence': float(conf),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })

            return detections

        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []