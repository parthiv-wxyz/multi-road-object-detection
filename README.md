# Multi-Road Object Detection using Enhanced YOLOv5s

## Overview

This project implements an enhanced version of **YOLOv5s** for multi-road object detection. The objective is to improve detection accuracy for objects of varying scales in complex road environments while maintaining real-time inference performance.

The model incorporates several architectural improvements over the original YOLOv5s:

- Lightweight Convolution Block (LWC3)
- Efficient Channel Attention (ECA)
- Bidirectional Feature Pyramid Network (BiFPN)
- Wise-IoU v3 Bounding Box Loss
- Optimized Anchor Generation
- Advanced Data Augmentation

The model is designed to detect multiple categories of road objects including:

- Traffic Signs
- Road Damage
- Vehicles
- Pedestrians
- Other road scene objects

---

# Project Structure

```
multi-road-object-detection/
│
├── configs/
├── datasets/
│   ├── idd/
│   ├── road_damage/
│   └── traffic_sign/
│
├── models/
│   ├── attention/
│   ├── fusion/
│   ├── losses/
│   ├── modules/
│   └── ...
│
├── scripts/
├── tests/
├── utils/
├── weights/
├── train.py
├── val.py
├── detect.py
├── requirements.txt
└── README.md
```

---

# Features

- Improved YOLOv5s backbone
- Lightweight architecture
- Multi-scale feature fusion
- Better small object detection
- Improved localization using Wise-IoU v3
- Optimized anchor generation
- Supports custom datasets
- GPU and CPU compatible
- Cross-platform support

---

# System Requirements

## Minimum

- Python 3.9+
- 8 GB RAM
- NVIDIA GPU (recommended)
- CUDA 11.8+ (optional)

## Recommended

- Python 3.10
- NVIDIA RTX GPU
- CUDA 11.8 or later
- 16 GB RAM

---

# Installation Guide

## 1. Clone Repository

```bash
git clone https://github.com/<username>/multi-road-object-detection.git

cd multi-road-object-detection
```

---

# Windows Installation

## Step 1

Install Python (3.9 or later)

https://www.python.org/downloads/

Verify

```bash
python --version
```

---

## Step 2

Create Virtual Environment

```bash
python -m venv .venv
```

Activate

Command Prompt

```cmd
.venv\Scripts\activate
```

PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

---

## Step 3

Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## Step 4

Install PyTorch

CPU

```bash
pip install torch torchvision torchaudio
```

GPU (CUDA 11.8)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Step 5

Install Project Requirements

```bash
pip install -r requirements.txt
```

---

# Linux Installation (Ubuntu)

## Step 1

Install Python

```bash
sudo apt update

sudo apt install python3 python3-pip python3-venv -y
```

---

## Step 2

Create Virtual Environment

```bash
python3 -m venv .venv
```

Activate

```bash
source .venv/bin/activate
```

---

## Step 3

Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## Step 4

Install PyTorch

CPU

```bash
pip install torch torchvision torchaudio
```

GPU (CUDA 11.8)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Step 5

Install Project Dependencies

```bash
pip install -r requirements.txt
```

---

# macOS Installation

## Step 1

Install Homebrew (if not installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

---

## Step 2

Install Python

```bash
brew install python
```

---

## Step 3

Create Virtual Environment

```bash
python3 -m venv .venv
```

Activate

```bash
source .venv/bin/activate
```

---

## Step 4

Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## Step 5

Install PyTorch

```bash
pip install torch torchvision torchaudio
```

---

## Step 6

Install Project Requirements

```bash
pip install -r requirements.txt
```

---

# Verify Installation

```bash
python -c "import torch; print(torch.__version__)"
```

GPU check

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected

```
True
```

or

```
False
```

if using CPU.

---

# Dataset Preparation

Organize datasets as:

```
datasets/

├── idd/
│   ├── images/
│   └── labels/
│
├── road_damage/
│   ├── images/
│   └── labels/
│
└── traffic_sign/
    ├── images/
    └── labels/
```

Update dataset YAML files accordingly.

---

# Training

Train from scratch

```bash
python train.py \
    --img 640 \
    --batch 16 \
    --epochs 300 \
    --data data/dataset.yaml \
    --cfg models/enhanced_yolov5s.yaml \
    --weights ""
```

Resume training

```bash
python train.py --resume
```

---

# Validation

```bash
python val.py \
    --weights runs/train/exp/weights/best.pt \
    --data data/dataset.yaml
```

---

# Inference

Image

```bash
python detect.py \
    --weights runs/train/exp/weights/best.pt \
    --source images/
```

Video

```bash
python detect.py \
    --weights runs/train/exp/weights/best.pt \
    --source video.mp4
```

Webcam

```bash
python detect.py \
    --weights runs/train/exp/weights/best.pt \
    --source 0
```

---

# Running Tests

Example

```bash
python tests/test_lwc3.py
```

---

# Model Improvements

The enhanced YOLOv5s introduces:

| Component | Purpose |
|-----------|----------|
| LWC3 | Lightweight feature extraction |
| ECA | Efficient channel attention |
| BiFPN | Multi-scale feature fusion |
| Wise-IoU v3 | Improved localization |
| Anchor Optimization | Better bounding box initialization |
| Data Augmentation | Improved generalization |

---

# Results

Example metrics:

| Metric | Baseline YOLOv5s | Enhanced YOLOv5s |
|---------|-----------------|------------------|
| Precision | - | - |
| Recall | - | - |
| mAP@0.5 | - | - |
| FPS | - | - |

(Update after training.)

---

# Citation

If you use this repository in your research, please cite the associated publication.

---

# License

This project is licensed under the MIT License.

---

# Acknowledgements

- Ultralytics YOLOv5
- PyTorch
- IDD Dataset
- RDD2022 / Road Damage Dataset
- Traffic Sign Datasets
- Open-source Computer Vision Community

---

# Contact

For questions or contributions, please open an issue or submit a pull request.