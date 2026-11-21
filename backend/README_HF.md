---
title: RoadSight Multi-Model Detection API
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# RoadSight Multi-Model Detection API

A FastAPI backend that runs **four specialised YOLOv5 object-detection models** in a single inference pipeline to analyse road scenes:

| Model | Task |
|-------|------|
| **IDD** | Indian Driving Dataset — vehicles, pedestrians, rickshaws, etc. |
| **RDD4D** | Road Damage Detection (4-class) — longitudinal, transverse, alligator, pothole |
| **RDD_22** | Road Damage Detection 2022 — extended damage taxonomy |
| **Traffic Sign** | Indian traffic sign detection + Stage-2 fine-grained classification (85 classes) |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info & model status |
| `GET` | `/health` | Health check |
| `GET` | `/models` | List loaded models, classes, colours |
| `POST` | `/detect` | Upload an image → returns detections + annotated image |

## Technical Notes

- **CPU-only** inference (PyTorch CPU) — expect ~2-5 s per image on the free tier.
- Total model weights ≈ 77 MB (bundled inside the Docker image).
- The Space sleeps after 48 h of inactivity; visiting the URL wakes it up.
