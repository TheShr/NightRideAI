"""
Configuration settings for NightRide AI
"""

import os
import torch
from typing import Dict, Any

class Config:
    """Configuration class for model settings and paths"""

    def __init__(self):
        # Model settings
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Model paths
        self.models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(self.models_dir, exist_ok=True)

        # Enhancement model
        self.enhancer_model = 'zero-dce'

        # Depth model
        self.depth_model = 'midas_dpt_large'

        # Object detection
        self.yolo_model = os.path.join(self.models_dir, 'yolov8n.pt')

        # Prefer custom trained YOLO best model if available
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        custom_best = os.path.join(repo_root, 'runs', 'detect', 'train2', 'weights', 'best.pt')
        if os.path.exists(custom_best):
            self.yolo_model = custom_best

        # Pothole detection
        self.pothole_model_path = os.path.join(self.models_dir, 'pothole_efficientnet.pth')

        # Alert settings
        self.alert_cooldown = 3.0  # seconds between alerts
        self.danger_distance = 5.0  # meters
        self.warning_distance = 10.0  # meters

        # Image processing
        self.target_size = (640, 480)

        # API settings
        self.host = '0.0.0.0'
        self.port = 8000

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for specific model"""
        configs = {
            'zero-dce': {
                'model_path': 'https://github.com/Li-Chongyi/Zero-DCE/raw/master/model.pth',
                'local_path': os.path.join(self.models_dir, 'zero_dce.pth')
            },
            'midas_dpt_large': {
                'model_type': 'DPT_Large',
                'local_path': os.path.join(self.models_dir, 'midas_dpt_large.pth')
            },
            'yolov8n': {
                'model_path': 'yolov8n.pt'
            }
        }
        return configs.get(model_name, {})