# Dataset Structure for NightRide AI

## Pothole Detection Dataset

```
datasets/
├── potholes/
│   ├── train/
│   │   ├── potholes/          # Images containing potholes
│   │   └── normal/            # Normal road images
│   └── val/
│       ├── potholes/
│       └── normal/
```

## Low-Light Enhancement Dataset

```
datasets/
├── lol_dataset/               # LOL Dataset for low-light enhancement
│   ├── train/
│   │   ├── low/               # Low-light images
│   │   └── high/              # Enhanced ground truth
│   └── val/
│       ├── low/
│       └── high/
```

## Data Sources

1. **Pothole Dataset**:
   - RDD2022: https://github.com/sekilab/RoadDamageDetector
   - Custom collection of pothole images

2. **LOL Dataset**:
   - LOL: https://daooshee.github.io/BMVC2018website/
   - Paired low/normal light images

## Preprocessing

- Resize all images to 224x224 for pothole detection
- Use original resolution for enhancement training
- Apply data augmentation (rotation, flip, brightness)
