# Demo Videos

This directory contains sample videos demonstrating NightRide AI in action.

## Available Videos

- `night_ride_demo.mp4` - Real-world night riding scenario
- `hazard_detection_demo.mp4` - Various hazard detection examples
- `low_light_enhancement.mp4` - Before/after enhancement comparison

## Recording Instructions

1. Record video at night/low-light conditions
2. Use phone camera with stabilization
3. Capture various scenarios:
   - Normal road
   - Vehicles ahead
   - Pedestrians
   - Potholes
   - Different lighting conditions

## Processing Demo Videos

Use the following command to process demo videos:

```bash
python backend/main.py --video demo_videos/night_ride_demo.mp4 --output results/
```
