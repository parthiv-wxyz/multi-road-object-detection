# 🚀 RoadSight — Deployment Guide

This guide walks you through deploying the **RoadSight Multi-Model Detection** stack:

| Component | Platform | Tier |
|-----------|----------|------|
| **Backend** (FastAPI + 4× YOLOv5 models) | Hugging Face Spaces (Docker) | Free |
| **Frontend** (React + Vite) | Vercel | Free |

---

## Prerequisites

- A free [Hugging Face](https://huggingface.co) account
- A free [Vercel](https://vercel.com) account
- [Git](https://git-scm.com/) and [Git LFS](https://git-lfs.com/) installed locally

---

## Part A — Deploy Backend to Hugging Face Spaces

### 1. Create a new Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose a name (e.g. `roadsight-api`)
3. Select **Docker** as the SDK
4. Visibility: **Public** (required for free tier)
5. Click **Create Space**

### 2. Clone the Space repo locally

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/roadsight-api
cd roadsight-api
```

### 3. Install Git LFS (for model weights)

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
```

> [!IMPORTANT]
> The model weights (`best.pt` files) total ~77 MB. Git LFS is **mandatory** —
> Hugging Face will reject pushes with large files otherwise.

### 4. Copy project files into the Space repo

From the root of this project (`YOLO/`), copy the following into the Space repo:

```bash
# From the YOLO project root:

# Dockerfile (goes to repo root)
cp backend/Dockerfile        <space-repo>/Dockerfile

# README for HF Spaces (must be named README.md)
cp backend/README_HF.md      <space-repo>/README.md

# App entrypoint (rename to app.py)
cp backend/app_hf.py         <space-repo>/backend/app.py

# Backend requirements
cp backend/requirements.txt  <space-repo>/backend/requirements.txt

# Pipeline utilities (entire folder)
cp -r project_utils/         <space-repo>/project_utils/

# YOLOv5 repo + trained weights (entire folder)
cp -r yolov5/                <space-repo>/yolov5/
```

Your Space repo should now look like:

```
<space-repo>/
├── Dockerfile
├── README.md              ← (was README_HF.md)
├── backend/
│   ├── app.py             ← (was app_hf.py)
│   └── requirements.txt
├── project_utils/
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── fusion.py
│   │   ├── orchestrator.py
│   │   ├── preprocessing.py
│   │   ├── sign_refiner.py
│   │   ├── spatial_filter.py
│   │   └── tiling.py
│   └── ...
└── yolov5/
    ├── hubconf.py
    ├── models/
    ├── utils/
    └── runs/
        ├── train/
        │   ├── idd_final/weights/best.pt
        │   ├── rdd4d_enhanced_final/weights/best.pt
        │   ├── enhanced_rdd2022_final/weights/best.pt
        │   └── traffic_sign_enhanced_1280_pretrained/weights/best.pt
        └── train-cls/
            └── traffic_sign_cls_50/weights/best.pt
```

### 5. Push to Hugging Face

```bash
cd <space-repo>
git add .
git commit -m "Initial deployment — RoadSight multi-model detection API"
git push
```

### 6. Wait for the build

- Go to your Space page on Hugging Face
- The **Logs** tab shows the Docker build progress
- First build takes **~5–10 minutes** (PyTorch CPU download is the bottleneck)
- Once the status turns **Running**, the API is live

### 7. Note your Space URL

Your API will be available at:

```
https://YOUR_USERNAME-roadsight-api.hf.space
```

Verify it works:

```bash
curl https://YOUR_USERNAME-roadsight-api.hf.space/health
```

Expected response:

```json
{
  "status": "ok",
  "models_loaded": ["IDD", "RDD4D", "RDD_22", "Traffic Sign"],
  "model_count": 4,
  "stage2_classifier_loaded": true
}
```

---

## Part B — Deploy Frontend to Vercel

### 1. Update the API URL

The frontend already has environment files set up:

- `frontend/.env.development` → points to `http://127.0.0.1:8000` (local dev)
- `frontend/.env.production` → needs your HF Spaces URL

Edit `frontend/.env.production`:

```env
VITE_API_URL=https://YOUR_USERNAME-roadsight-api.hf.space
```

### 2. Verify App.jsx uses the env variable

The frontend code should reference the API URL via the Vite env variable. In your
main component or API utility file, the base URL should be:

```javascript
const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
```

### 3. Push frontend to GitHub

```bash
cd frontend
git init   # if not already a git repo
git add .
git commit -m "Frontend — ready for Vercel deployment"
git remote add origin https://github.com/YOUR_USERNAME/roadsight-frontend.git
git push -u origin main
```

### 4. Deploy on Vercel

1. Go to [vercel.com/new](https://vercel.com/new)
2. Import your GitHub repository
3. Framework preset: **Vite**
4. Set the environment variable:
   - `VITE_API_URL` = `https://YOUR_USERNAME-roadsight-api.hf.space`
5. Click **Deploy**

Vercel will build and deploy in ~1–2 minutes. Your frontend will be live at a
`*.vercel.app` URL.

### 5. (Optional) Custom domain

Both Vercel and Hugging Face support custom domains if you want cleaner URLs.

---

## Important Notes

### Hugging Face Spaces — Free Tier Limits

| Resource | Limit |
|----------|-------|
| CPU | 2 vCPU |
| RAM | 16 GB |
| Disk | 50 GB |
| Inactivity timeout | **48 hours** (Space sleeps; visiting the URL wakes it) |

### Performance Expectations

- **CPU-only inference** — expect **~2–5 seconds per image** (all 4 models + Stage-2 classifier)
- First request after a cold start may take **~30–60 seconds** (model loading)
- Subsequent requests are faster as models stay in memory

### Model Weights Summary

| Model | Weight file | Size |
|-------|-------------|------|
| IDD | `idd_final/weights/best.pt` | 13.8 MB |
| RDD4D | `rdd4d_enhanced_final/weights/best.pt` | 16.7 MB |
| RDD_22 | `enhanced_rdd2022_final/weights/best.pt` | 16.9 MB |
| Traffic Sign | `traffic_sign_enhanced_1280_pretrained/weights/best.pt` | 21.8 MB |
| Traffic Sign Classifier | `traffic_sign_cls_50/weights/best.pt` | 8.3 MB |
| **Total** | | **~77 MB** |

### Keeping the Space Awake

If you need the Space to stay awake beyond 48 hours, you can:

1. **Upgrade to a paid Space** (Hugging Face Pro)
2. **Use an external ping service** (e.g., UptimeRobot) to hit `/health` periodically

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Space build fails | Check the Logs tab; most common issue is missing dependencies |
| CORS errors in browser | Verify `app_hf.py` has `allow_origins=["*"]` |
| Models not found | Check that `yolov5/runs/` was fully copied including all `best.pt` files |
| Git push rejected (file too large) | Install Git LFS and track `*.pt` files |
| Space shows "Sleeping" | Visit the Space URL to wake it; first request takes ~60s |
| Frontend can't reach backend | Verify `VITE_API_URL` matches your Space URL exactly |

---

## Local Development

For local development, everything works as before:

```bash
# Terminal 1 — Backend
cd backend
uvicorn app:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm run dev
```

The `.env.development` file ensures the frontend points to `http://127.0.0.1:8000` locally.
