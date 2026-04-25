"""
Logging configuration for NightRide AI
"""

import logging
import os
from datetime import datetime

def setup_logger():
    """Setup logging configuration"""
    # Create logs directory
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Log file path
    log_file = os.path.join(log_dir, f'nightride_{datetime.now().strftime("%Y%m%d")}.log')

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Reduce verbosity of some libraries
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('ultralytics').setLevel(logging.WARNING)