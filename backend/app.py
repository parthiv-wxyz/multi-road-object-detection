from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import torch
import io
import time
import base64

from pathlib import Path
from PIL import Image


app = FastAPI()


# Allow Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# PROJECT PATHS
# ==========================================

BACKEND_DIR = Path(__file__).resolve().parent

PROJECT_DIR = BACKEND_DIR.parent

YOLO_DIR = PROJECT_DIR / "yolov5"


MODEL_PATH = (
    YOLO_DIR
    / "runs"
    / "train"
    / "idd_final"
    / "weights"
    / "best.pt"
)


print("=" * 60)
print("YOLO DIRECTORY:")
print(YOLO_DIR)

print("\nMODEL PATH:")
print(MODEL_PATH)

print("\nMODEL EXISTS:")
print(MODEL_PATH.exists())
print("=" * 60)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = torch.hub.load(
    str(YOLO_DIR),
    "custom",
    path=str(MODEL_PATH),
    source="local"
)


# Detection confidence threshold
model.conf = 0.25


# ==========================================
# API ROUTES
# ==========================================

@app.get("/")
def home():

    return {
        "message": "YOLOv5 Road Object Detection API is running",

        "model": "idd_final",

        "model_path": str(MODEL_PATH),

        "model_exists": MODEL_PATH.exists()
    }


@app.post("/detect")
async def detect(
    file: UploadFile = File(...)
):

    # Read uploaded image
    image_bytes = await file.read()


    # Convert to PIL image
    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")


    # ==========================================
    # RUN YOLO INFERENCE
    # ==========================================

    start_time = time.time()

    results = model(image)

    inference_time = round(
        (time.time() - start_time) * 1000,
        2
    )


    # ==========================================
    # EXTRACT DETECTIONS
    # ==========================================

    detections = []

    predictions = results.pandas().xyxy[0]


    for _, row in predictions.iterrows():

        detections.append({

            "class": row["name"],

            "confidence": round(
                float(row["confidence"]) * 100,
                2
            ),

            "xmin": round(
                float(row["xmin"]),
                2
            ),

            "ymin": round(
                float(row["ymin"]),
                2
            ),

            "xmax": round(
                float(row["xmax"]),
                2
            ),

            "ymax": round(
                float(row["ymax"]),
                2
            )

        })


    # ==========================================
    # GENERATE IMAGE WITH BOUNDING BOXES
    # ==========================================

    rendered_image = results.render()[0]


    output_image = Image.fromarray(
        rendered_image
    )


    buffer = io.BytesIO()


    output_image.save(
        buffer,
        format="JPEG"
    )


    # Convert image to Base64
    encoded_image = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


    # ==========================================
    # RETURN RESULTS
    # ==========================================

    return {

        "image": encoded_image,

        "detections": detections,

        "total_objects": len(
            detections
        ),

        "inference_time_ms": inference_time

    }