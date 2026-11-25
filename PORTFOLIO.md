# Multi-Road Object Detection — Enhanced YOLOv5s

> **Domain**: Computer Vision · Deep Learning · Road Safety AI  
> **Stack**: Python · PyTorch · YOLOv5 · FastAPI · React · Vite  
> **Status**: Research & Deployment Complete

---

## 🗺️ Overview

A research-grade object detection system that extends **YOLOv5s** with custom architectural innovations to reliably detect road hazards, traffic signs, vehicles, and pedestrians in complex, unstructured road environments — particularly targeting Indian and global road conditions.

The system serves **three specialised detection models simultaneously** through a FastAPI backend and an interactive React frontend.

---

## 🎯 Problem Statement

Standard object detectors struggle with road scenes because:

- **~77.5%** of objects in unstructured road datasets (IDD) are small (< 32×32 px)
- Road damage (cracks, potholes) is **anisotropic and elongated**, unlike typical bounding-box targets
- Traffic signage varies wildly across **lighting conditions, viewing angles, and regional standards**
- Real-time inference demands conflict with accuracy requirements in safety-critical contexts

---

## 🏗️ Architecture & Innovations

The enhanced model introduces six custom components stacked on top of the YOLOv5s baseline:

| Component | Innovation | Impact |
|-----------|-----------|--------|
| **P2 Detection Head** | 160×160 feature map head (stride 4) for very small objects | +High spatial resolution for cracks, distant signs |
| **LWC3 / LWConv** | Multi-kernel grouped convolution (1×1, 3×3, 5×5, 7×7) replacing standard bottlenecks | Multi-scale receptive fields with controlled channel growth |
| **ECA Attention** | Channel attention via 1D Conv (adaptive kernel) — no dimensionality reduction | +Feature recalibration with only 40,969 extra params (+0.5%) |
| **AFP Neck** | Attention Feature Pyramid with channel + spatial attention replacing standard Concat | +Adaptive multi-scale feature fusion |
| **DirectionalConv** | Anisotropic filters (1×3, 3×1, 3×3) for road crack detection | +Captures longitudinal & transverse crack patterns |
| **Wise-IoU v3** | Dynamic non-monotonic focusing loss with distance attention | +Improved bounding-box localisation on hard samples |

### System Architecture

```
[ Input Image 640×640×3 ]
          │
  ┌───────┴───────┐
  │  ENHANCED     │
  │  BACKBONE     │
  │  LWC3 Blocks  │
  │  SPPF Module  │
  └───────┬───────┘
          │
  ┌───────┴───────┐
  │  NECK FUSION  │
  │  ConvECA      │
  │  AFP Pyramid  │
  │  DirectConv   │
  └───────┬───────┘
          │
  ┌───────┼──────────────┐
  ▼       ▼              ▼              ▼
P2:160×160  P3:80×80   P4:40×40    P5:20×20
Very Small  Small Obj  Medium Obj  Large Obj
(Cracks)   (Peds,Bike) (Cars)     (Buses)
```

---

## 📊 Results

### Model Complexity vs Baseline

| Architecture | Params | GFLOPs | CPU Latency | FPS |
|---|---|---|---|---|
| Baseline YOLOv5s | 7.24 M | 16.6 | 166.7 ms | ~6.0 |
| + P2 Head | 7.39 M | 20.2 | 237.1 ms | ~4.2 |
| + P2 + LWC3 | 7.89 M | 21.0 | 267.7 ms | ~3.7 |
| + P2 + LWC3 + ECA | 7.93 M | 21.1 | 274.4 ms | ~3.6 |
| **+ AFP (Full Enhanced)** | **9.59 M** | **28.9** | **355.0 ms** | **~2.8** |

### Detection Performance

| Dataset | Domain | Best mAP@0.5 | Precision | Recall |
|---|---|---|---|---|
| **IDD Final** | 15-class Indian road objects | **65.30%** | 82.16% | 57.84% |
| **RDD2022 Enhanced** | 4-class road damage (cracks) | **57.38%** | 61.09% | 55.69% |
| **Traffic Sign Enhanced** | 5-class traffic signs | **45.70%** | 69.94% | 38.12% |
| RDD4D Baseline | Road damage (clean split) | 45.45% | 46.11% | 44.76% |

> All models trained for 100 epochs with SGD (lr=0.01, momentum=0.937, weight decay=5e-4).

---

## 🧪 Key Research Findings

### What Worked
- **P2 head** significantly improves small object recall in dense IDD scenes
- **ECA attention** provides meaningful feature recalibration at virtually zero parameter cost
- **Batch size matters critically** — RDD4D mAP jumped from 39.29% → 57.38% simply by increasing batch size from 2 → 32, due to BatchNorm stability

### Bugs Discovered & Fixed
- **CutMix label leakage**: The original `CutMix` augmentation appended all labels from both images regardless of spatial overlap — causing the model to learn from "ghost" ground-truth boxes and destroying recall. Fixed with IoU-filtered label clipping.
- **WIoU objectness mismatch**: Bounding box regression used WIoU v3 loss, but the objectness branch was trained on CIoU targets — an inconsistency patched by synchronising both to WIoU targets.

### Trade-offs Documented
- AFP module adds +7.9 GFLOPs and +80ms latency per image — a significant overhead for edge deployment
- Precision consistently outpaced Recall by 25–50% across runs, indicating the model is conservative and misses hard/occluded objects

---

## 🖥️ System Stack

### Backend — FastAPI
- Loads **3 models concurrently** using `ThreadPoolExecutor`
- Colour-coded bounding boxes per model domain:
  - 🔵 `IDD` — Blue `#43A0FF`
  - 🟠 `RDD` Road Damage — Orange `#FFAA37`
  - 🟢 `Traffic Sign` — Green `#52D296`
- Deployed on Render with Docker

### Frontend — React + Vite
- Custom interactive **canvas-based image viewer**
- Zoom controls (1× to 4×) and panning
- **Toggleable model layers** (IDD / RDD / Traffic Sign)
- Confidence threshold slider
- Detection inspection panel

---

## 📁 Project Structure

```
multi-road-object-detection/
├── backend/           # FastAPI inference server (multi-model)
├── frontend/          # React + Vite UI with canvas viewer
├── datasets/          # IDD, RDD2022/RDD4D, Traffic Sign datasets
├── models/            # Custom modules: LWC3, ECA, AFP, DirectionalConv
│   ├── attention/     # ECA, ConvECA implementations
│   ├── fusion/        # AFP feature pyramid module
│   └── losses/        # Wise-IoU v3 loss function
├── experiments/       # Ablation study configs
├── notebooks/         # Training analysis and EDA notebooks
├── results/           # Confusion matrices, metric plots, training reports
├── scripts/           # Detection, export, and benchmark scripts
├── weights/           # Trained model weights (.pt files)
├── yolov5/            # Modified YOLOv5 codebase
├── train.py           # Main training entry point
├── val.py             # Validation script
└── detect.py          # Inference script
```

---

## 🚀 Datasets Used

| Dataset | Classes | Use Case |
|---------|---------|----------|
| **IDD** (India Driving Dataset) | 15 road categories | Unstructured Indian road scenes |
| **RDD2022 / RDD4D** (Road Damage Dataset) | 4 crack types | Global road damage detection |
| **Traffic Sign Dataset** | 5 sign categories | Traffic sign recognition |

---

## 🔧 Quickstart

```bash
# Clone & install
git clone https://github.com/<username>/multi-road-object-detection.git
cd multi-road-object-detection
pip install -r requirements.txt

# Train
python train.py --img 640 --batch 16 --epochs 100 \
  --data data/dataset.yaml --cfg models/enhanced_yolov5s.yaml

# Inference
python detect.py --weights weights/best.pt --source images/

# Start API
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

---

## 📌 Skills Demonstrated

- **Deep Learning & CV** — custom YOLOv5 architecture modifications, attention mechanisms, feature pyramid networks, IoU-based loss functions
- **Research Methodology** — ablation studies, empirical benchmarking, failure analysis
- **ML Engineering** — data augmentation pipeline, multi-dataset training, model optimisation
- **Full-Stack Deployment** — FastAPI + React frontend, Docker, Render cloud deployment
- **Code Quality** — bug identification (CutMix label leakage, WIoU mismatch), structured diagnostics and fixes

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
