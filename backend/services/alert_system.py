"""
Alert System
Handles voice alerts and notifications
"""

import time
import pyttsx3
import logging

logger = logging.getLogger(__name__)

class AlertSystem:
    """Voice alert system"""

    def __init__(self):
        self.engine = None
        self.last_alert_time = 0
        self.alert_cooldown = 3.0  # seconds
        self._init_tts()

    def _init_tts(self):
        """Initialize text-to-speech engine"""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 180)
            self.engine.setProperty('volume', 0.8)
            logger.info("TTS engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            self.engine = None

    def process_alerts(self, hazards: list) -> list:
        """Process hazards and generate alerts"""
        alerts = []

        current_time = time.time()

        for hazard in hazards:
            alert = self._generate_alert(hazard)
            if alert:
                alerts.append(alert)

                # Voice alert with cooldown
                if current_time - self.last_alert_time > self.alert_cooldown:
                    self._speak_alert(alert)
                    self.last_alert_time = current_time

        return alerts

    def _generate_alert(self, hazard: dict) -> dict:
        """Generate alert message for hazard"""
        try:
            hazard_type = hazard['type']
            level = hazard['level']

            if hazard_type == 'object':
                class_name = hazard['class']
                messages = {
                    'danger': f"Danger! {class_name} ahead",
                    'warning': f"Warning: {class_name} nearby"
                }
            elif hazard_type == 'pothole':
                messages = {
                    'warning': "Pothole detected"
                }
            else:
                return None

            message = messages.get(level)
            if message:
                return {
                    'message': message,
                    'level': level,
                    'timestamp': time.time()
                }

        except Exception as e:
            logger.error(f"Alert generation failed: {e}")

        return None

    def _speak_alert(self, alert: dict):
        """Speak alert message"""
        if self.engine and alert:
            try:
                self.engine.say(alert['message'])
                self.engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS failed: {e}")