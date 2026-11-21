# ============================================================
# Dockerfile — RoadSight Multi-Model Detection API
# Target: Hugging Face Spaces (Docker SDK, free-tier CPU)
# ============================================================

FROM python:3.11-slim

# --- System dependencies (OpenCV, git for torch.hub) --------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        git \
        fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies ------------------------------------
# Install PyTorch CPU-only first (largest layer, cached separately)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Copy and install requirements
COPY requirements_hf.txt /app/requirements_hf.txt
RUN pip install --no-cache-dir -r /app/requirements_hf.txt

# --- Application code & model weights -----------------------
# In the HF Space repo, files are at root level:
#   app.py, project_utils/, yolov5/, Dockerfile, README.md
COPY app.py              /app/app.py
COPY project_utils/      /app/project_utils/
COPY yolov5/             /app/yolov5/

# --- Environment & networking -------------------------------
ENV HF_HOME=/tmp/hf_home
ENV PYTHONUNBUFFERED=1
EXPOSE 7860

# --- Non-root user (HF Spaces requirement) ------------------
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p /tmp/hf_home && chown appuser:appuser /tmp/hf_home
USER appuser

# --- Entrypoint ---------------------------------------------
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
