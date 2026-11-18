from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import base64
import io
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
YOLO_DIR = PROJECT_DIR / "yolov5"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from project_utils.pipeline import MultiRoadDetectionPipeline, DEFAULT_MODEL_PROFILES


app = FastAPI(title="Multi-Model Road Scene Detection API")


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODEL PATHS & VISUAL COLORS
# ============================================================

MODEL_CONFIG = {
    "IDD": os.getenv(
        "IDD_MODEL_PATH",
        str(YOLO_DIR / "runs" / "train" / "idd_final" / "weights" / "best.pt"),
    ),
    "RDD4D": os.getenv(
        "RDD4D_MODEL_PATH",
        str(YOLO_DIR / "runs" / "train" / "rdd4d_enhanced_final" / "weights" / "best.pt"),
    ),
    "RDD_22": os.getenv(
        "RDD2022_MODEL_PATH",
        str(YOLO_DIR / "runs" / "train" / "enhanced_rdd2022_final" / "weights" / "best.pt"),
    ),
    "Traffic Sign": os.getenv(
        "TRAFFIC_SIGN_MODEL_PATH",
        str(
            YOLO_DIR
            / "runs"
            / "train"
            / "traffic_sign_enhanced_1280_pretrained"
            / "weights"
            / "best.pt"
        ),
    ),
}

TRAFFIC_SIGN_CLS_PATH = Path(
    os.getenv(
        "TRAFFIC_SIGN_CLS_PATH",
        str(YOLO_DIR / "runs" / "train-cls" / "traffic_sign_cls_50" / "weights" / "best.pt"),
    )
)

# Distinct overlay colors per model domain
MODEL_COLORS = {
    "IDD": (67, 160, 255),          # Electric Blue
    "RDD4D": (255, 170, 55),        # Amber / Orange
    "RDD_22": (235, 87, 87),        # Coral / Crimson
    "Traffic Sign": (82, 210, 150), # Mint Green
}


# ============================================================
# MODEL LOADING & PIPELINE INITIALIZATION
# ============================================================

def resolve_model_path(path_string: str) -> Path:
    return Path(path_string).expanduser().resolve()


def load_model(model_name: str, path_string: str):
    path = resolve_model_path(path_string)

    if not path.exists():
        raise FileNotFoundError(
            f"{model_name} model was not found:\n{path}\n\n"
            f"Set the corresponding environment variable if your run directory differs."
        )

    print("=" * 70)
    print(f"Loading {model_name}")
    print(f"Path: {path}")

    model = torch.hub.load(
        str(YOLO_DIR),
        "custom",
        path=str(path),
        source="local",
    )
    model.eval()

    print(f"{model_name}: loaded")
    print("=" * 70)
    return model


MODELS = {
    name: load_model(name, path)
    for name, path in MODEL_CONFIG.items()
}

PIPELINE = MultiRoadDetectionPipeline(
    models=MODELS,
    yolo_dir=YOLO_DIR,
    classifier_weights=TRAFFIC_SIGN_CLS_PATH,
    profiles=DEFAULT_MODEL_PROFILES,
)


# ============================================================
# RENDERING HELPERS
# ============================================================

def get_font(size: int = 16):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_all_detections(image: Image.Image, detections):
    """
    Draw detections from all models onto one combined image with clear
    color-coded bounding boxes and readable labels.
    """
    output = image.copy()
    draw = ImageDraw.Draw(output)
    label_font = get_font(16)

    for detection in detections:
        model_name = detection["model"]
        color = MODEL_COLORS.get(model_name, (255, 255, 255))

        x1 = int(detection["xmin"])
        y1 = int(detection["ymin"])
        x2 = int(detection["xmax"])
        y2 = int(detection["ymax"])

        confidence = detection["confidence"]
        class_name = detection["class"]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = f"{model_name} | {class_name} {confidence:.1f}%"
        bbox = draw.textbbox((0, 0), label, font=label_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        label_x = min(max(0, x1), max(0, output.width - text_w - 12))
        label_y = max(0, y1 - text_h - 8)

        draw.rectangle(
            [label_x, label_y, label_x + text_w + 10, label_y + text_h + 6],
            fill=color,
        )
        # Use white text on darker red background, dark ink on bright backgrounds
        text_fill = (255, 255, 255) if model_name == "RDD_22" else (10, 20, 30)
        draw.text(
            (label_x + 5, label_y + 3),
            label,
            fill=text_fill,
            font=label_font,
        )

    return output


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "YOLOv5 Multi-Model Road Object Detection API is running",
        "models": {
            name: {
                "path": str(resolve_model_path(path)),
                "exists": resolve_model_path(path).exists(),
                "native_img_size": DEFAULT_MODEL_PROFILES[name].img_size,
                "native_conf": DEFAULT_MODEL_PROFILES[name].conf,
            }
            for name, path in MODEL_CONFIG.items()
        },
        "stage2_traffic_sign_classifier": PIPELINE.sign_refiner.is_ready,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": list(MODELS.keys()),
        "model_count": len(MODELS),
        "stage2_classifier_loaded": PIPELINE.sign_refiner.is_ready,
    }


@app.get("/models")
def get_models():
    model_info = {}
    for name, model in MODELS.items():
        classes = (
            list(model.names.values())
            if isinstance(model.names, dict)
            else list(model.names)
        )
        color = MODEL_COLORS.get(name, (255, 255, 255))
        prof = DEFAULT_MODEL_PROFILES[name]
        model_info[name] = {
            "classes": classes,
            "class_count": len(classes),
            "color_rgb": color,
            "color_hex": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
            "img_size": prof.img_size,
            "confidence": prof.conf,
        }
    return {
        "models": model_info,
        "stage2_classifier_classes": len(PIPELINE.sign_refiner.names),
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    try:
        result = PIPELINE.predict(image)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Detection pipeline failed: {type(exc).__name__}: {exc}",
        )

    rendered_image = render_all_detections(image, result["detections"])
    encoded_image = image_to_base64(rendered_image)

    return {
        "image": encoded_image,
        "detections": result["detections"],
        "total_objects": result["total_objects"],
        "inference_time_ms": result["inference_time_ms"],
        "models": result["models"],
        "per_model": result["per_model"],
        "pipeline_modules": result["pipeline_modules"],
        "filtered_summary": result["filtered_summary"],
        "image_width": image.width,
        "image_height": image.height,
    }
