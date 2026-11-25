# Multi-Road Object Detection using Enhanced YOLOv5s

> **A multi-domain road perception system** combining traffic-scene object detection (IDD), road damage & crack detection (RDD2022), and traffic sign recognition — all served through a unified FastAPI + React inference pipeline.

---

## Overview

This project implements an **enhanced version of YOLOv5s** fine-tuned across three distinct Indian road datasets. The base architecture is extended with:

| Module | Description |
|--------|-------------|
| **LWC3 / LWConv** | Lightweight multi-kernel grouped convolutions replacing standard 3×3 bottlenecks |
| **ECA (Efficient Channel Attention)** | Parameter-efficient channel recalibration without dimensionality reduction |
| **AFP (Attention Feature Pyramid)** | Adaptive multi-scale feature fusion with channel + spatial attention |
| **DirectionalConv** | Anisotropic 1×3 / 3×1 / 3×3 filters for road crack detection |
| **P2 Detection Head** | 160×160 extra-resolution head for very small objects (cracks, distant signs) |
| **Wise-IoU v3** | Dynamic non-monotonic focusing bounding-box regression loss |
| **Optimized Anchors** | Percentile-based anchor clustering per dataset |

Six independently trained YOLOv5s model checkpoints are integrated into the project:

| Model | Dataset | Training Run | Task |
|-------|---------|--------------|------|
| **IDD** | Indian Driving Dataset (IDD) | `idd_final` | Road-scene object detection |
| **RDD4D** | RDD4D | `rdd4d_enhanced_final` | Road damage detection |
| **RDD2022 – Original** | RDD-2022 (Kaggle) | `enhanced_rdd2022_finalmod` | Road damage & crack detection |
| **RDD2022_mod – Updated** | RDD-2022 (Kaggle) | `enhanced_rdd2022_final` | Road damage & crack detection |
| **Traffic Sign – Original** | Indian Traffic Signs (Kaggle) | `traffic_sign_enhanced_1280_pretrainedmod` | Traffic sign detection |
| **Traffic Sign_mod – Updated** | Indian Traffic Signs (Kaggle) | `traffic_sign_enhanced_1280_pretrained` | Traffic sign detection |

Each model has independently trained `best.pt` weights. The backend configuration supports selecting the appropriate checkpoint through environment variables.

---

## Datasets

### 1. Indian Driving Dataset Detections (YOLOv11 format)
Used for training the road-scene object detector

- **Source**: [Kaggle — Indian Driving Dataset Detections (YOLOv11) by redzapdos123](https://www.kaggle.com/datasets/redzapdos123/indian-driving-dataset-detections-yolov11)
- Used for: anchor statistics, bounding-box distribution analysis, pre-labelled sample verification

---

### 2. RDD-2022 — Road Damage Detection Dataset
Used for training the road damage detector (`enhanced_rdd2022_final`).

- **Source**: [Kaggle — RDD-2022 by aliabdelmenam](https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022)
- **Classes**: `Alligator crack`, `Longitudinal crack`, `Transverse crack`, `Pothole`, `Other corruption`
- **Challenge**: Thin, elongated, low-contrast cracks; severe class imbalance (potholes underrepresented); scale variation from dashcam perspectives

---

### 3. Indian Traffic Signs Dataset
Used for training the traffic sign category detector (`traffic_sign_enhanced_1280_pretrained`).

- **Source**: [Kaggle — Traffic Signs Dataset (Indian Roads) by kaustubhrastogi17](https://www.kaggle.com/datasets/kaustubhrastogi17/traffic-signs-dataset-indian-roads)
- **Classes**: `Regulatory`, `Mandatory`, `Informatory`, `General`, `Warning`
- **Challenge**: High intra-class visual diversity; illumination variance; Informatory class is severely underrepresented (~55 instances vs ~160 for Warning)

---

## Project Structure

```
YOLO/
│
├── backend/                        # FastAPI inference server
│   ├── app.py                      # Main API (multi-model parallel inference)
│   ├── app_hf.py                   # Hugging Face Spaces variant
│   ├── app_render.py               # Render.com deployment variant
│   ├── Dockerfile                  # Container build file
│   ├── requirements.txt            # Backend dependencies
│   ├── requirements_hf.txt         # HF Spaces dependencies
│   └── requirements_render.txt     # Render dependencies
│
├── frontend/                       # React + Vite web interface
│   └── src/
│       ├── App.jsx                 # Main component (canvas viewer, model toggles)
│       ├── App.css                 # Styles
│       └── assets/                 # Static assets
│
├── project_utils/                  # Custom Python utilities
│   ├── augmentations/
│   │   ├── cutmix.py               # CutMix augmentation (see known bug)
│   │   └── __init__.py
│   ├── losses/
│   │   └── wise_iou.py             # Wise-IoU implementation
│   ├── pipeline/
│   │   ├── orchestrator.py         # MultiRoadDetectionPipeline
│   │   ├── preprocessing.py        # Image pre-processing
│   │   ├── sign_refiner.py         # Stage-2 traffic sign classifier
│   │   ├── spatial_filter.py       # Overlap / NMS filtering
│   │   ├── tiling.py               # High-res tile inference
│   │   ├── fusion.py               # Detection fusion across models
│   │   └── __init__.py
│   └── refined_anchor.py           # Percentile-based anchor optimizer
│
├── scripts/                        # Training, evaluation & utility scripts
│   ├── train_baseline.py           # Train baseline YOLOv5s
│   ├── train_enhanced.py           # Train enhanced model variants
│   ├── validate_model.py           # Validation script
│   ├── validate_enhanced.py        # Full validation w/ PR curves + F1
│   ├── detect_model.py             # Image detection script
│   ├── detect_video.py             # Video stream inference
│   ├── webcam.py                   # Live webcam inference
│   ├── export.py                   # ONNX / TensorRT exporter
│   ├── benchmark.py                # FPS / FLOPs / latency benchmark
│   ├── ablation.py                 # Ablation experiment runner
│   ├── gradcam.py                  # Grad-CAM visualization
│   ├── dataset_statistics.py       # Dataset stats
│   ├── visualize_dataset.py        # Dataset visualizer
│   ├── generate_anchors.py         # Custom anchor generator
│   ├── finetune_models.py          # Fine-tuning helper
│   ├── split_traffic_sign_dataset.py    # Train/val/test splitter
│   ├── convert_rdd4d_json_to_yolo.py    # JSON → YOLO label converter
│   ├── verify_dataset.py           # Dataset integrity checker
│   └── prepare_tuned_datasets.py   # Dataset preparation pipeline
│
├── weights/                        # Trained model weights
│   ├── idd.pt                      # IDD road-scene detector
│   ├── rdd2022.pt                  # RDD2022 road damage detector
│   ├── rdd4d.pt                    # RDD4D road damage detector
│   └── traffic_sign.pt             # Traffic sign detector
│
├── results/                        # Training reports, figures, metrics
│   ├── IDD_Final_Training_Report.md
│   ├── enhanced_rdd2022_final_report.md
│   ├── rdd4d_enhanced_final_report.md
│   ├── traffic_sign_enhanced_1280_pretrained.md
│   ├── confusion_matrix/
│   ├── figures/
│   ├── metrics/
│   └── all_dataset_image_report.csv
│
├── datasets/                       # Local dataset copies
│   ├── idd/
│   ├── road_damage/
│   └── traffic_sign/
│
├── tests/                          # Unit tests
├── yolov5/                         # YOLOv5 framework (modified)
├── requirements.txt                # Project dependencies
├── render.yaml                     # Render.com deployment config
├── DEPLOYMENT.md                   # Deployment guide
└── README.md
```

---

## Features

### Architecture Enhancements

- **P2 Extra-Resolution Detection Head** — adds a 160×160 feature map (stride 4) on top of the standard P3/P4/P5 heads, enabling detection of objects smaller than 16×16 px
- **LWC3 / LWConv Blocks** — multi-kernel grouped convolutions (1×1, 3×3, 5×5, 7×7 kernels in 4 channel groups) replace standard bottlenecks at P4/P5 depth
- **ECA Efficient Channel Attention** — 1D convolution over pooled features with adaptive kernel size; adds only ~41K parameters
- **AFP Attention Feature Pyramid** — replaces standard Concat in neck with channel attention → spatial attention → 3×3 refinement at each scale
- **DirectionalConv** — horizontal (1×3) + vertical (3×1) + standard (3×3) branch fusion for crack feature extraction
- **Wise-IoU v3 Loss** — dynamic focusing function suppresses outlier gradients and emphasizes normal-quality anchors

### Multi-Domain Inference Pipeline

- **Parallel multi-model inference** using `ThreadPoolExecutor` — IDD, RDD2022, and Traffic Sign models run concurrently on the same image
- **Stage-2 traffic sign classifier** (`sign_refiner.py`) re-classifies YOLO detections using a dedicated YOLOv5-cls head
- **Spatial overlap filter** (`spatial_filter.py`) suppresses redundant cross-model bounding boxes
- **Tile-based high-resolution inference** (`tiling.py`) for large image inputs
- **Color-coded bounding box rendering**: IDD (Electric Blue `#43A0FF`), RDD4D (Amber `#FFAA37`), RDD2022 (Coral `#EB5757`), Traffic Sign (Mint Green `#52D296`)

### Web Interface (React + Vite)

- Drag-and-drop image upload
- Interactive canvas viewer with **zoom (1×–4×)** and **panning**
- **Toggleable model layers** — show/hide IDD / RDD / Traffic Sign detections independently
- **Confidence threshold slider** per model
- Detection inspection panel with class name, confidence, and bounding-box coordinates

### Training Infrastructure

- Percentile-based anchor optimization (`generate_anchors.py`, `refined_anchor.py`)
- Dataset preparation pipeline — JSON → YOLO label conversion, train/val/test splitting, duplicate label removal
- Full ablation framework across 4 component axes: P2 head, LWC3, ECA, AFP
- Dataset visualization, statistics, and class distribution analysis

---

## System Requirements

### Minimum

| Requirement | Specification |
|-------------|--------------|
| Python | 3.9+ |
| RAM | 8 GB |
| GPU | NVIDIA GPU (optional, CPU supported) |
| CUDA | 11.8+ (optional) |
| Disk | ~5 GB (weights + datasets) |

### Recommended

| Requirement | Specification |
|-------------|--------------|
| Python | 3.10 |
| GPU | NVIDIA RTX 2060 or better |
| CUDA | 11.8 or later |
| RAM | 16 GB |

---

## Installation Guide

### 1. Clone Repository

```bash
git clone https://github.com/parthiv-wxyz/multi-road-object-detection.git
cd multi-road-object-detection
```

---

### Windows

```powershell
# Step 1 — Verify Python
python --version

# Step 2 — Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1      # PowerShell
# OR
.venv\Scripts\activate.bat      # Command Prompt

# Step 3 — Upgrade pip
python -m pip install --upgrade pip

# Step 4 — Install PyTorch (CPU)
pip install torch torchvision torchaudio

# Step 4 — Install PyTorch (GPU / CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Step 5 — Install project requirements
pip install -r requirements.txt
```

---

### Linux (Ubuntu)

```bash
sudo apt update && sudo apt install python3 python3-pip python3-venv -y
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

### macOS

```bash
brew install python
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

---

### Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## Dataset Preparation

Download datasets from Kaggle and organize as follows:

```
datasets/
├── idd/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
│
├── road_damage/              # RDD-2022
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
│
└── traffic_sign/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

---

## Training

### Train IDD Road-Scene Detector

```bash
cd yolov5

python train.py \
    --img 640 \
    --batch 16 \
    --epochs 100 \
    --data ../model_changes/data/idd.yaml \
    --cfg ../model_changes/models/yolov5s_p2_lwc3_eca_afp.yaml \
    --weights yolov5s.pt \
    --name idd_final \
    --hyp data/hyps/hyp.scratch-low.yaml
```

### Train Road Damage Detector (RDD2022)

```bash
python train.py \
    --img 640 \
    --batch 32 \
    --epochs 100 \
    --data ../model_changes/data/road_damage.yaml \
    --cfg ../model_changes/models/enhanced_rdd4d.yaml \
    --weights yolov5s.pt \
    --name enhanced_rdd2022_final
```

> **Important**: Use `--batch 32` or higher for RDD. Batch size 2 causes BatchNorm instability and positive-to-negative anchor ratio exceeding 1:15,000 with the P2 head.

### Train Traffic Sign Detector

```bash
python train.py \
    --img 1280 \
    --batch 16 \
    --epochs 100 \
    --data ../model_changes/data/traffic_sign.yaml \
    --cfg ../model_changes/models/traffic_sign_enhanced.yaml \
    --weights yolov5s.pt \
    --name traffic_sign_enhanced_1280_pretrained
```

> **Note**: 1280×1280 input resolution is used for traffic signs to preserve detail for small and distant sign instances.

### Resume Training

```bash
python train.py --resume runs/train/<exp_name>/weights/last.pt
```

### Gradient Accumulation (low-VRAM)

```bash
python train.py --batch 4 --accumulate 8   # effective batch = 32
```

---

## Validation

```bash
# Validate IDD model
python val.py \
    --weights runs/train/idd_final/weights/best.pt \
    --data ../model_changes/data/idd.yaml \
    --img 640

# Validate RDD2022 model
python val.py \
    --weights runs/train/enhanced_rdd2022_final/weights/best.pt \
    --data ../model_changes/data/road_damage.yaml \
    --img 640

# Validate Traffic Sign model
python val.py \
    --weights runs/train/traffic_sign_enhanced_1280_pretrained/weights/best.pt \
    --data ../model_changes/data/traffic_sign.yaml \
    --img 1280
```

Full validation with PR curves and F1 metrics:

```bash
python scripts/validate_enhanced.py --weights weights/idd.pt --data model_changes/data/idd.yaml
```

---

## Inference

### Single Image

```bash
python scripts/detect_model.py \
    --weights weights/idd.pt \
    --source path/to/image.jpg \
    --conf 0.35 \
    --img 640
```

### Image Directory

```bash
python yolov5/detect.py \
    --weights weights/idd.pt \
    --source path/to/images/ \
    --conf 0.35
```

### Multi-Model Parallel Inference (Pipeline)

```bash
# Start the FastAPI backend
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The pipeline runs IDD + RDD2022 + Traffic Sign models concurrently and returns merged detections with color-coded bounding boxes.

**Model-specific recommended confidence thresholds:**

| Model | Recommended Confidence | F1-Optimal Threshold |
|-------|----------------------|---------------------|
| IDD | 0.35 | ~0.326 |
| RDD2022 | 0.20 | ~0.194 |
| Traffic Sign | 0.43 | ~0.431 |

### Export to ONNX / TensorRT

```bash
python scripts/export.py \
    --weights weights/idd.pt \
    --img 640 \
    --batch 1 \
    --format onnx
```

---

## Running the Web Application

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

API endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info & model paths |
| `/health` | GET | Health check & loaded model list |
| `/models` | GET | Per-model classes, colors, config |
| `/detect` | POST | Upload image → merged detection result |

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Access at `http://localhost:5173`

---

## Running Tests

```bash
# Test lightweight convolution block
python tests/test_lwc3.py

# Test AFP fusion module
python tests/test_afp.py

# Benchmark FPS / latency
python scripts/benchmark.py --weights weights/idd.pt --img 640
```

---

## Model Architecture Details

### Architecture Variants & Complexity

All benchmarks: single-thread CPU, input `1×3×640×640`.

| Variant | Config | Layers | Parameters | GFLOPs | CPU Latency | FPS |
|---------|--------|--------|-----------|--------|-------------|-----|
| Baseline YOLOv5s | `yolov5s.yaml` | 125 | 7.24 M | 16.6 | 166.7 ms | ~6.0 |
| + P2 Head | `yolov5s_p2.yaml` | 153 | 7.39 M | 20.2 | 237.1 ms | ~4.2 |
| + P2 + LWC3 | `yolov5s_p2_lwc3.yaml` | 213 | 7.89 M | 21.0 | 267.7 ms | ~3.7 |
| + P2 + LWC3 + ECA | `yolov5s_p2_lwc3_eca.yaml` | 232 | 7.93 M | 21.1 | 274.4 ms | ~3.6 |
| **Full Enhanced** | `yolov5s_p2_lwc3_eca_afp.yaml` | 304 | 9.59 M | 28.9 | 355.0 ms | ~2.8 |
| Traffic Sign (nc=5) | `traffic_sign_enhanced.yaml` | 310 | 9.96 M | 28.0 | 327.5 ms | ~3.1 |
| RDD4D (nc=4) | `enhanced_rdd4d.yaml` | 238 | 8.30 M | 20.2 | 245.1 ms | ~4.1 |

### Module Descriptions

#### P2 Detection Head (160×160)

Standard YOLOv5s uses a minimum stride of 8 (80×80 at P3). Objects smaller than 16×16 px lose spatial resolution before reaching the detection head. The P2 head adds stride-4 feature maps (160×160), producing 76,800 anchor boxes per image. This directly benefits:
- Thin hairline cracks in road surfaces
- Distant traffic signs
- Small pedestrians and cyclists in the far field

#### LWC3 / LWConv Block

Replaces standard 3×3 bottleneck convolutions at P4/P5 depth:

```
Input Channels C
│
├─ Group 1 (C/4): 1×1 Conv
├─ Group 2 (C/4): 3×3 Conv
├─ Group 3 (C/4): 5×5 Conv (dilated)
└─ Group 4 (C/4): 7×7 Conv (dilated)
│
Concat → BatchNorm → SiLU → 1×1 Pointwise Conv
```

Provides multi-scale receptive field diversity while keeping parameter growth controlled.

#### ECA Efficient Channel Attention

Avoids the information bottleneck in Squeeze-and-Excitation (SE) networks:

```
x → AdaptiveAvgPool2D → Conv1D(k) → Sigmoid → ⊗ x
```

Adaptive kernel size `k = ψ(C)` is determined by channel dimension. Adds only **40,969 parameters** across the full model.

#### AFP — Attention Feature Pyramid Fusion

Replaces standard Concat in the FPN/PAN neck:

1. Spatial alignment (bilinear interpolation if needed)
2. Channel concatenation (2C → C via 1×1 Conv + BN + SiLU)
3. Channel attention (GlobalAvgPool → MLP → Sigmoid)
4. Spatial attention (7×7 Conv on avg+max pooled maps → Sigmoid)
5. Refinement (3×3 Conv + BN + SiLU)

Adds ~1.7 M parameters and ~7.9 GFLOPs. For edge deployment, see the BiFPN replacement recommendation.

#### DirectionalConv (Road Crack Specific)

```
Input
├─ 1×3 Horizontal Conv (longitudinal cracks)
├─ 3×1 Vertical Conv (transverse cracks)
└─ 3×3 Standard Conv
│
1×1 Fusion Conv
```

Specifically designed to capture anisotropic crack patterns on asphalt.

#### Wise-IoU v3 Loss

Dynamic non-monotonic focusing loss:

```
β = L_IoU / L̄_IoU        (outlier degree vs. running mean)
r = β / (δ · α^(β−δ))    (dynamic focusing weight)
L_WIoUv3 = r · L_WIoUv1   (final weighted loss)
```

Normal-quality anchors receive the highest gradient. Very-high-quality (converged) and very-low-quality (outlier) anchors are down-weighted to focus training on productive samples.

---

## Results

### Model 1 — IDD Road-Scene Detector

**Dataset**: Indian Driving Dataset (IDD) · **7 classes** · **100 epochs** · **Batch 16** · **640×640**

| Metric | Baseline YOLOv5s (paper) | Current IDD Model | Improvement |
|--------|--------------------------|-------------------|-------------|
| Precision | 60.00% | **82.16%** | +22.16 pp (+37%) |
| Recall | 35.00% | **57.84%** | +22.84 pp (+65%) |
| mAP@0.5 | 28.00% | **65.30%** (ep 93) | +37.30 pp (+133%) |
| mAP@0.5:0.95 | — | **43.03%** | — |
| Best F1 | — | **~0.68** @ conf 0.326 | — |

**Class-wise AP@0.5:**

| Class | AP@0.5 | Confusion Matrix |
|-------|--------|-----------------|
| person | 51.1% | 0.47 |
| car | 71.7% | 0.66 |
| autorickshaw | 74.0% | 0.67 |
| truck | 66.9% | 0.62 |
| **bus** | **75.2%** | **0.67** |
| motorcycle | 69.6% | 0.67 |
| bicycle | 48.5% | 0.48 |
| **All classes** | **65.3%** | — |

**Training loss (final epoch 99):**

| Loss | Train | Validation |
|------|-------|-----------|
| Box | 0.0371 | 0.0365 |
| Objectness | 0.0316 | 0.0230 |
| Classification | 0.0078 | 0.0103 |

**Key findings**: Strong performance on large vehicles (bus, autorickshaw, car). Smaller objects (bicycle, person) remain more challenging due to high object density and occlusion in Indian road scenes.

---

### Model 2 — RDD2022 Road Damage Detector

**Dataset**: RDD-2022 (Kaggle) · **5 classes** · **100 epochs** · **Batch 32** · **640×640**

| Metric | Value |
|--------|-------|
| Precision | **61.09%** |
| Recall | **55.69%** |
| mAP@0.5 | **57.38%** (ep 99) |
| mAP@0.5:0.95 | **28.94%** |
| Best F1 | **~0.58** @ conf 0.194 |

**Class-wise AP@0.5:**

| Class | AP@0.5 | Confusion Matrix |
|-------|--------|-----------------|
| Other corruption | **73.0%** | 0.70 |
| Alligator crack | **64.6%** | 0.56 |
| Longitudinal crack | 53.0% | 0.49 |
| Transverse crack | 52.3% | 0.42 |
| Pothole | 44.0% | 0.37 |
| **All classes** | **57.4%** | — |

**Training loss (final epoch):**

| Loss | Train | Validation |
|------|-------|-----------|
| Box | 0.0346 | 0.0411 |
| Objectness | 0.0047 | 0.0029 |
| Classification | 0.0042 | 0.0087 |

**Key findings**: Most balanced precision/recall of the three models (61% / 56%). Potholes remain the weakest class due to irregular shapes and appearance variation. The model achieves stable convergence at batch size 32 (previous batch=2 experiments showed a 6.2% mAP drop due to BatchNorm instability).

---

### Model 3 — Traffic Sign Detector

**Dataset**: Indian Traffic Signs (Kaggle) · **5 classes** · **100 epochs** · **Batch 16** · **1280×1280**

| Metric | Value | Best |
|--------|-------|------|
| Precision | **89.25%** | 100.00% |
| Recall | **77.65%** | 78.32% |
| mAP@0.5 | **80.19%** | 80.30% (ep ~95) |
| mAP@0.5:0.95 | **45.86%** | 47.17% |
| Best F1 | **~0.80** @ conf 0.431 | — |

**Class-wise AP@0.5:**

| Class | AP@0.5 | Confusion Matrix | Assessment |
|-------|--------|-----------------|------------|
| **Mandatory** | **95.3%** | 0.98 | Excellent |
| **Regulatory** | **93.5%** | 0.91 | Excellent |
| General | 84.9% | 0.80 | Good |
| Warning | 73.3% | 0.77 | Moderate |
| Informatory | 50.2% | 0.42 | Weak |
| **All classes** | **79.5%** | — | Good |

**Training loss (final epoch):**

| Loss | Train | Validation |
|------|-------|-----------|
| Box | 0.028 | 0.032 |
| Objectness | 0.0057 | 0.0043 |
| Classification | 0.0035 | 0.0041 |

**Key findings**: Highest performing model across all three. 1280×1280 resolution is essential for capturing small and distant traffic signs. Informatory class (only ~55 instances vs ~160 for other classes) has significantly lower AP due to class imbalance. Duplicate detections observed near Mandatory signs — tune NMS IoU ≈ 0.45 at inference.

---

### Ablation Study — Component Contributions (IDD Dataset)

| Configuration | mAP@0.5 | Δ vs. Baseline |
|---------------|---------|---------------|
| Baseline YOLOv5s | 43.26% | — |
| + P2 Head | ~48% | +4.7 pp |
| + P2 + LWC3 | ~52% | +8.7 pp |
| + P2 + LWC3 + ECA | ~54% | +10.7 pp |
| + P2 + LWC3 + ECA + AFP (Full) | **65.30%** | +22 pp |

---

### Cross-Model Summary

| Model | Dataset | mAP@0.5 | Precision | Recall | F1 | FPS (CPU) |
|-------|---------|---------|-----------|--------|----|-----------|
| IDD Detector | IDD (7 cls) | **65.30%** | 82.16% | 57.84% | 0.68 | ~2.8 |
| RDD2022 Detector | RDD-2022 (5 cls) | **57.38%** | 61.09% | 55.69% | 0.58 | ~4.1 |
| Traffic Sign Detector | Traffic Signs (5 cls) | **80.19%** | 89.25% | 77.65% | 0.80 | ~3.1 |

---

## Known Issues & Diagnostics

### 1. CutMix Augmentation Bug

The CutMix implementation in `project_utils/augmentations/cutmix.py` has a ground-truth label corruption bug: all labels from `img2` are appended to the mixed image regardless of whether they fall inside the pasted patch. This creates phantom ground-truth annotations, suppresses recall, and trains the model to make false predictions.

**Impact**: Estimated −3 to −5% recall across all models.

**Fix**: Filter bounding boxes to only include those with ≥20% area overlap with the actual pasted crop region.

### 2. Wise-IoU Implementation Mismatch

Two Wise-IoU implementations exist in the codebase:
- `project_utils/losses/wise_iou.py` — a power-scaled focal IoU (not true WIoU v3)
- `yolov5/models/losses/wise_iou.py` — the correct WIoU v3 with distance attention and dynamic focusing

Additionally, in `yolov5/utils/loss.py`, the objectness target is computed with CIoU while the box regression uses WIoU — a loss objective mismatch.

### 3. Batch Size Sensitivity (P2 Head)

The P2 head generates 76,800 anchor locations per image (160×160×3). At batch size ≤ 4:
- Positive-to-negative anchor ratio exceeds 1:15,000
- BatchNorm running statistics become unstable
- WIoU v3 running mean fluctuates, causing over-suppression of hard samples

**Solution**: Use `--batch 32` minimum, or `--batch 4 --accumulate 8` for effective batch 32.

### 4. AFP Latency Overhead

The AFP module accounts for the bulk of the latency increase (+80ms vs ECA-only). On CPU-only deployment targets, consider replacing AFP with a lightweight Weighted BiFPN neck to recover ~45% FPS improvement.

---

## Model Export

```bash
# Export to ONNX
python scripts/export.py --weights weights/idd.pt --img 640 --format onnx

# Export to TensorRT (requires TensorRT installed)
python scripts/export.py --weights weights/idd.pt --img 640 --format engine --half

# Export to OpenVINO (INT8)
python scripts/export.py --weights weights/idd.pt --img 640 --format openvino
```

Expected speedup from quantization:

| Format | Speedup vs PyTorch FP32 | mAP Drop |
|--------|------------------------|----------|
| ONNX FP32 | ~1.5× | <0.1% |
| TensorRT FP16 | ~2.5× | <0.3% |
| TensorRT INT8 | ~3.8× | <0.8% |
| OpenVINO INT8 | ~2.5× | <0.8% |

---

## Deployment

### Docker

```bash
cd backend
docker build -t road-detection-api .
docker run -p 8000:8000 \
    -e IDD_MODEL_PATH=/weights/idd.pt \
    -e RDD2022_MODEL_PATH=/weights/rdd2022.pt \
    -e TRAFFIC_SIGN_MODEL_PATH=/weights/traffic_sign.pt \
    -v $(pwd)/weights:/weights \
    road-detection-api
```

### Render.com

See [DEPLOYMENT.md](DEPLOYMENT.md) and [render.yaml](render.yaml) for one-click Render deployment.

### Hugging Face Spaces

See `backend/app_hf.py` and `backend/README_HF.md`.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `IDD_MODEL_PATH` | `yolov5/runs/train/idd_final/weights/best.pt` | Path to IDD model |
| `RDD2022_MODEL_PATH` | `yolov5/runs/train/enhanced_rdd2022_final/weights/best.pt` | Path to RDD2022 model |
| `TRAFFIC_SIGN_MODEL_PATH` | `yolov5/runs/train/traffic_sign_enhanced_1280_pretrained/weights/best.pt` | Path to Traffic Sign model |
| `TRAFFIC_SIGN_CLS_PATH` | `yolov5/runs/train-cls/traffic_sign_cls_50/weights/best.pt` | Stage-2 classifier path |

---

## Citation

If you use this repository in your research, please cite the associated publication and the following datasets:

```
@dataset{rdd2022_kaggle,
  author  = {aliabdelmenam},
  title   = {RDD-2022 Road Damage Detection Dataset},
  year    = {2022},
  url     = {https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022}
}

@dataset{traffic_signs_indian,
  author  = {kaustubhrastogi17},
  title   = {Traffic Signs Dataset — Indian Roads},
  url     = {https://www.kaggle.com/datasets/kaustubhrastogi17/traffic-signs-dataset-indian-roads}
}

@dataset{idd_yolo_detections,
  author  = {redzapdos123},
  title   = {Indian Driving Dataset Detections (YOLOv11)},
  url     = {https://www.kaggle.com/datasets/redzapdos123/indian-driving-dataset-detections-yolov11}
}
```

---

## License

This project is licensed under the **MIT License**.

---

## Acknowledgements

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [PyTorch](https://pytorch.org)
- [RDD-2022 Dataset (Kaggle)](https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022)
- [Indian Traffic Signs Dataset (Kaggle)](https://www.kaggle.com/datasets/kaustubhrastogi17/traffic-signs-dataset-indian-roads)
- [Indian Driving Dataset Detections YOLOv11 (Kaggle)](https://www.kaggle.com/datasets/redzapdos123/indian-driving-dataset-detections-yolov11)
- [IDD — Indian Driving Dataset](https://idd.insaan.iiit.ac.in/)
- ECA-Net, BiFPN (EfficientDet), Wise-IoU v3 papers
- Open-source Computer Vision Community

---

## Contact

For questions, bug reports, or contributions, please open an issue or submit a pull request.
