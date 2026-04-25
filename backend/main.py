"""
NightRide AI - Main FastAPI Application
Low-Light Monocular Depth Estimation and Hazard Detection
"""

import asyncio
import base64
import binascii
import cv2
import io
import json
import logging
import os
import time
from typing import Dict, List, Optional
import numpy as np
from PIL import Image
import pyttsx3
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import websockets

from models.enhancer import LowLightEnhancer
from models.depth_estimator import DepthEstimator
from models.object_detector import ObjectDetector
from models.pothole_detector import PotholeDetector
from services.hazard_engine import HazardEngine
from services.alert_system import AlertSystem
from utils.config import Config
from utils.logger import setup_logger

# Setup logging
setup_logger()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NightRide AI",
    description="Low-Light Hazard Detection for Two-Wheelers",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
config = Config()
enhancer = None
depth_estimator = None
object_detector = None
pothole_detector = None
hazard_engine = None
alert_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global enhancer, depth_estimator, object_detector, pothole_detector, hazard_engine, alert_system

    logger.info("Initializing models...")

    try:
        enhancer = LowLightEnhancer()
        depth_estimator = DepthEstimator()
        object_detector = ObjectDetector()
        pothole_detector = PotholeDetector()
        hazard_engine = HazardEngine()
        alert_system = AlertSystem()

        logger.info("All models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

class PredictRequest(BaseModel):
    image: str


def decode_base64_image(image_base64: str) -> Optional[np.ndarray]:
    if not image_base64:
        return None

    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except binascii.Error as e:
        logger.error(f"Base64 decode failed: {e}")
        return None

    if not image_bytes:
        return None

    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


@app.post("/predict")
async def predict_image(request: PredictRequest):
    """Process single image for hazard detection"""
    try:
        # Decode base64 image from JSON payload
        image = decode_base64_image(request.image)
        if image is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image data"})

        # Process pipeline
        start_time = time.time()

        # 1. Enhance image
        enhanced = enhancer.enhance(image)

        # 2. Estimate depth
        depth_map = depth_estimator.estimate_depth(enhanced)

        # 3. Detect objects
        detections = object_detector.detect(enhanced)

        # 4. Detect potholes
        potholes = pothole_detector.detect(enhanced)

        # 5. Analyze hazards
        hazards = hazard_engine.analyze_hazards(detections, potholes, depth_map, image.shape)

        # 6. Generate alerts
        alerts = alert_system.process_alerts(hazards)

        processing_time = time.time() - start_time

        # Convert images to base64 for response
        _, enhanced_buffer = cv2.imencode('.jpg', enhanced)
        enhanced_b64 = base64.b64encode(enhanced_buffer).decode()

        # Create depth heatmap
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        _, depth_buffer = cv2.imencode('.jpg', depth_colored)
        depth_b64 = base64.b64encode(depth_buffer).decode()

        return {
            "enhanced_image": enhanced_b64,
            "depth_map": depth_b64,
            "detections": detections,
            "potholes": potholes,
            "hazards": hazards,
            "alerts": alerts,
            "processing_time": processing_time,
            "fps": 1.0 / processing_time if processing_time > 0 else 0
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# WebSocket for real-time video processing
active_connections = []

@app.websocket("/video_stream")
async def video_stream(websocket: WebSocket):
    """Real-time video stream processing"""
    await websocket.accept()
    active_connections.append(websocket)

    try:
        while True:
            # Receive frame data
            data = await websocket.receive_text()
            frame_data = json.loads(data)

            # Decode base64 image
            image_data = base64.b64decode(frame_data['image'])
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Process frame (same pipeline as predict)
            enhanced = enhancer.enhance(image)
            depth_map = depth_estimator.estimate_depth(enhanced)
            detections = object_detector.detect(enhanced)
            potholes = pothole_detector.detect(enhanced)
            hazards = hazard_engine.analyze_hazards(detections, potholes, depth_map, image.shape)
            alerts = alert_system.process_alerts(hazards)

            # Send results back
            response = {
                "detections": detections,
                "potholes": potholes,
                "hazards": hazards,
                "alerts": alerts
            }

            await websocket.send_json(response)

    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )