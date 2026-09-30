from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import base64
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont


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
# PROJECT PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
YOLO_DIR = PROJECT_DIR / "yolov5"

# You can override these from the environment if your run names differ.
MODEL_CONFIG = {
    "IDD": os.getenv(
        "IDD_MODEL_PATH",
        str(YOLO_DIR / "runs" / "train" / "idd_final" / "weights" / "best.pt"),
    ),
    "RDD4D": os.getenv(
        "RDD4D_MODEL_PATH",
        str(
            YOLO_DIR
            / "runs"
            / "train"
            / "rdd4d_enhanced_final"
            / "weights"
            / "best.pt"
        ),
    ),
    "RDD_22": os.getenv(
        "RDD4D_MODEL_PATH",
        str(
            YOLO_DIR
            / "runs"
            / "train"
            / "enhanced_rdd2022_final"
            / "weights"
            / "best.pt"
        ),
    ),
    "Traffic Sign": os.getenv(
        "TRAFFIC_SIGN_MODEL_PATH",
        str(
            YOLO_DIR
            / "runs"
            / "train"
            / "traffic_sign_enhanced_final"
            / "weights"
            / "best.pt"
        ),
    ),
}

CONFIDENCE = float(os.getenv("YOLO_CONF", "0.25"))
IOU = float(os.getenv("YOLO_IOU", "0.45"))
IMAGE_SIZE = int(os.getenv("YOLO_IMG_SIZE", "640"))

# Different line colors make overlapping predictions from the three
# independent models distinguishable in the combined output image.
MODEL_COLORS = {
    "IDD": (67, 160, 255),
    "RDD4D": (255, 170, 55),
    "Traffic Sign": (82, 210, 150),
}


# ============================================================
# HELPERS
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

    model.conf = CONFIDENCE
    model.iou = IOU
    model.eval()

    print(f"{model_name}: loaded")
    print("=" * 70)

    return model


# Load all three models once when the API starts.
MODELS = {
    name: load_model(name, path)
    for name, path in MODEL_CONFIG.items()
}


def extract_detections(model_name, results, model):
    detections = []

    predictions = results.xyxy[0]

    for prediction in predictions:
        x1, y1, x2, y2, confidence, class_id = prediction.tolist()

        class_id = int(class_id)

        if isinstance(model.names, dict):
            class_name = model.names.get(class_id, str(class_id))
        else:
            class_name = model.names[class_id]

        detections.append({
            "model": model_name,
            "class": str(class_name),
            "confidence": round(float(confidence) * 100, 2),
            "xmin": round(float(x1), 2),
            "ymin": round(float(y1), 2),
            "xmax": round(float(x2), 2),
            "ymax": round(float(y2), 2),
        })

    return detections


def run_one_model(model_name, model, image):
    start = time.perf_counter()

    with torch.inference_mode():
        results = model(image, size=IMAGE_SIZE)

    elapsed = round((time.perf_counter() - start) * 1000, 2)
    detections = extract_detections(model_name, results, model)

    return {
        "model": model_name,
        "detections": detections,
        "count": len(detections),
        "inference_time_ms": elapsed,
    }


def get_font(size=16):
    # Use a normal system font when available. PIL's default is the fallback.
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


def render_all_detections(image, detections):
    """
    Draw detections from all three models onto one image.

    This is intentionally done outside YOLO's result.render() so the
    predictions of all three independent models can be shown together.
    """
    output = image.copy()
    draw = ImageDraw.Draw(output)

    label_font = get_font(16)
    small_font = get_font(13)

    for detection in detections:
        model_name = detection["model"]
        color = MODEL_COLORS.get(model_name, (255, 255, 255))

        x1 = int(detection["xmin"])
        y1 = int(detection["ymin"])
        x2 = int(detection["xmax"])
        y2 = int(detection["ymax"])

        confidence = detection["confidence"]
        class_name = detection["class"]

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=color,
            width=3,
        )

        label = f"{model_name} | {class_name} {confidence:.1f}%"

        bbox = draw.textbbox((0, 0), label, font=label_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        label_x = x1
        label_y = max(0, y1 - text_h - 8)

        draw.rectangle(
            [label_x, label_y, label_x + text_w + 10, label_y + text_h + 6],
            fill=color,
        )
        draw.text(
            (label_x + 5, label_y + 3),
            label,
            fill=(10, 20, 30),
            font=label_font,
        )

    # Small legend in the top-left corner.
    # legend_x = 14
    # legend_y = 14
    # legend_h = 30
    # legend_w = 185

    # draw.rounded_rectangle(
    #     [
    #         legend_x,
    #         legend_y,
    #         legend_x + legend_w,
    #         legend_y + legend_h * len(MODEL_COLORS) + 8,
    #     ],
    #     radius=6,
    #     fill=(10, 18, 31, 225),
    # )

    # y = legend_y + 7
    # for model_name, color in MODEL_COLORS.items():
    #     draw.rectangle([legend_x + 8, y + 3, legend_x + 20, y + 15], fill=color)
    #     draw.text(
    #         (legend_x + 28, y),
    #         model_name,
    #         fill=(240, 245, 250),
    #         font=small_font,
    #     )
    #     y += legend_h

    return output


def image_to_base64(image):
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
            }
            for name, path in MODEL_CONFIG.items()
        },
        "confidence": CONFIDENCE,
        "iou": IOU,
        "image_size": IMAGE_SIZE,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": list(MODELS.keys()),
        "model_count": len(MODELS),
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

    # --------------------------------------------------------
    # Run all three models concurrently on the same image.
    # --------------------------------------------------------
    overall_start = time.perf_counter()

    model_results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_one_model, name, model, image.copy()): name
            for name, model in MODELS.items()
        }

        for future in as_completed(futures):
            model_name = futures[future]
            try:
                model_results.append(future.result())
            except Exception as exc:
                import traceback

                print("\n" + "=" * 80)
                print(f"ERROR IN MODEL: {model_name}")
                print("=" * 80)
                traceback.print_exc()
                print("=" * 80 + "\n")

                raise HTTPException(
                    status_code=500,
                    detail=f"{model_name} inference failed: {type(exc).__name__}: {exc}",
                )

    total_time = round((time.perf_counter() - overall_start) * 1000, 2)

    # Keep a stable model order in the response.
    model_order = list(MODELS.keys())
    model_results.sort(key=lambda item: model_order.index(item["model"]))

    all_detections = []
    for result in model_results:
        all_detections.extend(result["detections"])

    rendered_image = render_all_detections(image, all_detections)
    encoded_image = image_to_base64(rendered_image)

    per_model = {
        result["model"]: {
            "count": result["count"],
            "inference_time_ms": result["inference_time_ms"],
        }
        for result in model_results
    }

    return {
        "image": encoded_image,
        "detections": all_detections,
        "total_objects": len(all_detections),
        "inference_time_ms": total_time,
        "models": model_results,
        "per_model": per_model,
        "image_width": image.width,
        "image_height": image.height,
    }
