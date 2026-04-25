# NightRide AI

Low-light monocular depth estimation and hazard detection for two-wheelers. A real-time smart riding assistant that combines multiple state-of-the-art computer vision models to enhance safety during night and low-visibility conditions using a single camera.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

---

## Overview

NightRide AI addresses a critical gap in two-wheeler safety: the absence of affordable, real-time hazard detection in low-light conditions. Using only a monocular camera, the system performs image enhancement, depth estimation, object detection, and pothole identification — all within a sub-100ms inference loop on GPU.

This project integrates four pretrained SOTA models into a unified pipeline backed by a FastAPI WebSocket server and a live React dashboard.

---

## Features

**Low-Light Enhancement**
Zero-DCE++ adaptively brightens frames without overexposure, preserving spatial detail for downstream model accuracy.

**Monocular Depth Estimation**
MiDaS DPT Large infers per-pixel relative depth from a single image, enabling distance-based hazard thresholding without stereo hardware.

**Edge-Guided Refinement**
Canny edge detection sharpens object boundaries in the depth map, improving near-obstacle segmentation under poor lighting.

**Object Detection**
YOLOv8n detects persons, vehicles, and obstacles in real time with bounding boxes and confidence scores.

**Pothole Detection**
A fine-tuned EfficientNet-B0 classifier identifies road surface anomalies trained on a labeled pothole dataset.

**Hazard Alerting**
A rule-based hazard engine triggers voice alerts when detected objects fall within critical depth thresholds.

**Real-Time Dashboard**
A React frontend streams the live enhanced feed, depth map overlay, detection annotations, and hazard analytics simultaneously.

---

## Architecture

```
Camera Feed
    |
    v
Zero-DCE++ Enhancement
    |
    +----------------+------------------+
    |                |                  |
    v                v                  v
MiDaS Depth     YOLOv8n Detection   EfficientNet Pothole
    |                |                  |
    v                v                  v
Edge Refinement  Bounding Boxes     Surface Label
    |                |                  |
    +----------------+------------------+
                     |
                     v
             Hazard Engine (Rule-Based)
                     |
                     v
         FastAPI WebSocket Backend
                     |
                     v
           React Dashboard (Live)
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- CUDA-compatible GPU (optional, CPU inference supported)

### Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Run Both Servers

```bash
./run.sh
```

---

## Usage

1. Connect or configure your camera input in `backend/config.py`.
2. Start the application using `./run.sh`.
3. Open the dashboard in your browser at `http://localhost:5173`.
4. The live feed, depth map, and detection overlays stream automatically via WebSocket.
5. Hazard alerts are triggered verbally when an obstacle is detected within the configured depth threshold.

---

## Models

All models are downloaded automatically on first run and cached locally. Both CPU and GPU inference are supported.

| Model | Purpose | Source |
|---|---|---|
| Zero-DCE++ | Low-light image enhancement | Custom pretrained |
| MiDaS DPT Large | Monocular depth estimation | Intel ISL |
| YOLOv8n | Object detection | Ultralytics |
| EfficientNet-B0 | Pothole classification | Fine-tuned on pothole dataset |

---

## Results

> Screenshots and qualitative results — enhanced frames, depth map visualizations, and detection overlays — will be added here.

---

## Future Work

- IMU sensor fusion for metric-scale depth correction
- Multi-camera stereo depth as a higher-accuracy alternative
- Deep learning-based lane detection module
- Optimized video streaming for reduced frontend latency
- Mobile application deployment for Android

---

