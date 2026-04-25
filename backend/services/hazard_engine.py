"""
Hazard Analysis Engine
Analyzes detections and depth to identify hazards
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

class HazardEngine:
    """Hazard analysis and alerting engine"""

    def __init__(self):
        self.danger_distance = 5.0  # meters
        self.warning_distance = 10.0  # meters

    def analyze_hazards(self, detections: list, potholes: list, depth_map: np.ndarray, image_shape: tuple) -> list:
        """Analyze potential hazards"""
        hazards = []

        # Analyze object detections
        for detection in detections:
            hazard = self._analyze_detection(detection, depth_map, image_shape)
            if hazard:
                hazards.append(hazard)

        # Analyze potholes
        for pothole in potholes:
            hazard = self._analyze_pothole(pothole, depth_map, image_shape)
            if hazard:
                hazards.append(hazard)

        return hazards

    def _analyze_detection(self, detection: dict, depth_map: np.ndarray, image_shape: tuple) -> dict:
        """Analyze single detection for hazard potential"""
        try:
            bbox = detection['bbox']
            class_name = detection['class']

            # Calculate center of bounding box
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2

            # Get depth at center (assuming depth_map is normalized 0-1 or actual meters)
            # For simplicity, assume depth_map values are in meters
            if center_y < depth_map.shape[0] and center_x < depth_map.shape[1]:
                depth = depth_map[center_y, center_x]
            else:
                depth = 20.0  # Default far distance

            # Determine if object is in path (center region)
            image_width = image_shape[1]
            center_region = (image_width * 0.3, image_width * 0.7)

            in_path = center_x >= center_region[0] and center_x <= center_region[1]

            # Classify hazard level
            hazard_level = 'none'
            if depth < self.danger_distance and in_path:
                hazard_level = 'danger'
            elif depth < self.warning_distance and in_path:
                hazard_level = 'warning'

            if hazard_level != 'none':
                return {
                    'type': 'object',
                    'class': class_name,
                    'bbox': bbox,
                    'depth': float(depth),
                    'in_path': in_path,
                    'level': hazard_level
                }

        except Exception as e:
            logger.error(f"Detection analysis failed: {e}")

        return None

    def _analyze_pothole(self, pothole: dict, depth_map: np.ndarray, image_shape: tuple) -> dict:
        """Analyze pothole for hazard potential"""
        try:
            bbox = pothole['bbox']

            # Calculate center
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2

            # Potholes are typically on the ground, so check if in path
            image_width = image_shape[1]
            center_region = (image_width * 0.2, image_width * 0.8)  # Wider for potholes

            in_path = center_x >= center_region[0] and center_x <= center_region[1]

            if in_path:
                return {
                    'type': 'pothole',
                    'bbox': bbox,
                    'in_path': in_path,
                    'level': 'warning'
                }

        except Exception as e:
            logger.error(f"Pothole analysis failed: {e}")

        return None