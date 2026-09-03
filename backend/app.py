from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pathlib import Path

import torch
import io
from PIL import Image
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent
YOLO_DIR = BASE_DIR.parent / "yolov5"

MODEL_PATH = YOLO_DIR / "runs" / "enhanced" / "weights" / "best.pt"

model = torch.hub.load(
    str(YOLO_DIR),
    "custom",
    path=str(MODEL_PATH),
    source="local"
)

model.conf = 0.25


@app.get("/")
def home():
    return {
        "message": "YOLOv5 Road Object Detection API is running"
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    results = model(image)

    rendered_image = results.render()[0]

    output_image = Image.fromarray(
        rendered_image
    )

    buffer = io.BytesIO()

    output_image.save(
        buffer,
        format="JPEG"
    )

    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="image/jpeg"
    )