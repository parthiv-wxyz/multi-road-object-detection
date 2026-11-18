import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

print("=" * 60)
print("Testing Backend API & Multi-Stage Detection Pipeline")
print("=" * 60)

# 1. Test /health
r_health = client.get("/health")
assert r_health.status_code == 200
data_health = r_health.json()
print("Health Status:", data_health["status"])
print("Models Loaded:", data_health["models_loaded"])
print("Stage-2 Sign Classifier Loaded:", data_health["stage2_classifier_loaded"])
assert data_health["model_count"] == 4
assert data_health["stage2_classifier_loaded"] is True

# 2. Test /models
r_models = client.get("/models")
assert r_models.status_code == 200
data_models = r_models.json()
print("\nLoaded Model Profiles:")
for name, info in data_models["models"].items():
    print(
        f"  [{name}] {info['class_count']} classes | "
        f"imgsz={info['img_size']} | conf={info['confidence']} | Color={info['color_hex']}"
    )
assert data_models["models"]["Traffic Sign"]["img_size"] == 1280
assert data_models["models"]["Traffic Sign"]["confidence"] >= 0.40

# 3. Test /detect end-to-end on multiple road scenes (upload-only, zero manual knobs)
test_images = [
    ROOT / "tests" / "sample1.jpg",
    ROOT / "tests" / "sample2.jpg",
    ROOT / "tests" / "sampled.jpg",
    ROOT / "frontend" / "src" / "assets" / "sample-road.jpg",
]

for img_path in test_images:
    with open(img_path, "rb") as f:
        files = {"file": (img_path.name, f, "image/jpeg")}
        r_detect = client.post("/detect", files=files)

    assert r_detect.status_code == 200, f"Failed on {img_path.name}: {r_detect.text}"
    res = r_detect.json()
    print(f"\nImage: {img_path.name} ({res['image_width']}x{res['image_height']})")
    print(f"  Total objects : {res['total_objects']} | Latency: {res['inference_time_ms']} ms")
    print(f"  Per model     : {res['per_model']}")
    print(f"  Filtered stats: {res['filtered_summary']}")
    for d in res["detections"]:
        print(f"    - [{d['model']}] {d['class']:<24} {d['confidence']:5.1f}%")

    # On these pure traffic/vehicle scenes, Traffic Sign must have ZERO false positives on vehicles
    sign_count = res["per_model"]["Traffic Sign"]["count"]
    assert sign_count == 0, f"Unexpected false-positive traffic sign on {img_path.name}: {res['detections']}"

print("\nAll Backend API & Pipeline tests passed successfully!")
