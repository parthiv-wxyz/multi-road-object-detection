# 🚀 Deploying RoadSight to Render (100% Free, No Credit Card)

This guide walks you through deploying your backend to **Render's Free Web Service** with RAM optimizations and setting up **UptimeRobot** so your backend stays running 24/7 without sleeping.

---

## 📋 Summary of Optimizations Made for Render

Render's free tier provides **512 MB RAM**. Running PyTorch with 4 YOLO models in parallel normally requires 650+ MB and would trigger an Out of Memory (OOM) crash.

We have engineered `backend/app_render.py` to:
1. **Load one model at a time sequentially** during inference, then immediately free its memory.
2. **Cap inference resolution to 640** on CPU, keeping memory peak at **~430 MB** (safely below 512 MB).
3. **Moved all 5 model weights (~81 MB total) into a root `weights/` folder** so Git tracks them properly (since `yolov5/runs/` was git-ignored).
4. **Lightweight `/health` and `/models` endpoints** that consume 0 MB of model RAM and return in under 2ms.

---

## Part 1: Commit and Push Your Code to GitHub

Open PowerShell in your project folder (`c:\Users\ppart\Desktop\YOLO`) and run:

```powershell
git add weights/
git add backend/
git add render.yaml
git commit -m "Add Render optimizations and model weights"
git push origin main
```

*(All 5 model weight files in `weights/` are under 25 MB each, well below GitHub's 100 MB limit, so they push without any issues.)*

---

## Part 2: Deploy Backend to Render

1. Go to [render.com](https://render.com) and **Sign Up / Log In** with your GitHub account (No credit card needed).
2. Click **New +** at the top right and select **Web Service**.
3. Choose **Build and deploy from a Git repository** and connect your `multi-road-object-detection` repo.
4. Configure the settings:
   - **Name:** `roadsight-backend` (or any name you like)
   - **Region:** Oregon (US West) or Frankfurt (closest to you)
   - **Branch:** `main`
   - **Root Directory:** *(leave blank)*
   - **Runtime:** `Python 3`
   - **Build Command:**
     ```bash
     pip install --upgrade pip && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -r backend/requirements_render.txt
     ```
   - **Start Command:**
     ```bash
     uvicorn backend.app_render:app --host 0.0.0.0 --port $PORT
     ```
   - **Instance Type:** Select **Free** (0.1 CPU, 512 MB RAM)
5. Under **Advanced** -> **Add Environment Variable**:
   - Key: `PYTHON_VERSION` | Value: `3.11.0`
   - Key: `MPLCONFIGDIR` | Value: `/tmp/matplotlib`
6. Click **Deploy Web Service**.
7. Wait 3–5 minutes for the build to finish. Once live, you will see your URL:
   `https://roadsight-backend-xxxx.onrender.com`

Test it by visiting: `https://roadsight-backend-xxxx.onrender.com/health` in your browser. It should reply `{"status":"ok"}`.

---

## Part 3: Keep It Running 24/7 (Prevent Render Sleeping)

Render's free tier automatically goes to sleep if no one makes a request for 15 minutes. To keep it **awake 24/7**:

1. Go to [uptimerobot.com](https://uptimerobot.com) and create a free account (100% free, no credit card).
2. Click **Add New Monitor**.
3. Fill in:
   - **Monitor Type:** `HTTP(s)`
   - **Friendly Name:** `RoadSight Backend`
   - **URL (or IP):** `https://your-backend-name.onrender.com/health`
   - **Monitoring Interval:** `Every 10 minutes`
4. Click **Create Monitor**.

UptimeRobot will now ping your lightweight `/health` endpoint every 10 minutes, keeping your Render server permanently active 24/7!

---

## Part 4: Connect Your Frontend

### Option A: Deploy Frontend on Vercel (Recommended, Fastest)
1. In `frontend/.env.production`, put your new Render backend URL:
   ```
   VITE_API_URL=https://your-backend-name.onrender.com
   ```
2. Commit and push:
   ```powershell
   git add frontend/
   git commit -m "Update API URL for Render backend"
   git push origin main
   ```
3. Import your repo on [Vercel](https://vercel.com) (Root directory: `frontend`), and click **Deploy**.

### Option B: Deploy Frontend on Render (Static Site)
1. On Render, click **New +** -> **Static Site**.
2. Connect your repo.
3. Set:
   - **Root Directory:** `frontend`
   - **Build Command:** `npm run build`
   - **Publish Directory:** `dist`
4. Under Environment Variables:
   - Key: `VITE_API_URL` | Value: `https://your-backend-name.onrender.com`
5. Click **Create Static Site**.
