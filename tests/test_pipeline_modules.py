"""
Unit tests for all zero-retraining inference pipeline enhancement modules.
"""

from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_utils.pipeline import (
    AdaptiveIlluminationNormalizer,
    CrossDomainSpatialFilter,
    RoadROITiler,
    RoadSurfaceEnhancer,
    VehicleConflictResolver,
    WeightedBoxesFuser,
)

print("=" * 65)
print("PIPELINE ENHANCEMENT MODULES UNIT TESTS")
print("=" * 65)

# 1. AdaptiveIlluminationNormalizer
normalizer = AdaptiveIlluminationNormalizer()
dark_img = Image.fromarray(np.full((240, 320, 3), 30, dtype=np.uint8))
brightened = normalizer(dark_img)
assert np.mean(np.asarray(brightened)) > np.mean(np.asarray(dark_img)), "Dark image should brighten"

normal_img = Image.fromarray(np.full((240, 320, 3), 130, dtype=np.uint8))
unchanged = normalizer(normal_img)
assert np.array_equal(np.asarray(unchanged), np.asarray(normal_img)), "Well-lit image should remain unchanged"
print("[PASS] AdaptiveIlluminationNormalizer")

# 2. RoadSurfaceEnhancer (LAB-CLAHE)
enhancer = RoadSurfaceEnhancer()
grad = np.tile(np.linspace(60, 140, 320, dtype=np.uint8), (240, 1))
road_img = Image.fromarray(np.stack([grad, grad, grad], axis=-1))
enhanced_road = enhancer(road_img)
assert enhanced_road.size == road_img.size and enhanced_road.mode == "RGB"
print("[PASS] RoadSurfaceEnhancer")

# 3. RoadROITiler
tiler = RoadROITiler()
hi_res = Image.fromarray(np.zeros((720, 1280, 3), dtype=np.uint8))
tiles = tiler.generate_tiles(hi_res)
assert len(tiles) == 2, f"Expected 2 road ROI tiles, got {len(tiles)}"
projected = tiler.project_detections(
    [{"model": "RDD_22", "class": "pothole", "confidence": 65.0, "xmin": 10.0, "ymin": 20.0, "xmax": 110.0, "ymax": 90.0}],
    x_offset=tiles[1][1],
    y_offset=tiles[1][2],
    full_width=1280,
    full_height=720,
)
assert projected[0]["xmin"] == 10.0 + tiles[1][1]
assert projected[0]["ymin"] == 20.0 + tiles[1][2]
small_img = Image.fromarray(np.zeros((400, 500, 3), dtype=np.uint8))
assert len(tiler.generate_tiles(small_img)) == 0
print("[PASS] RoadROITiler")

# 4. VehicleConflictResolver
resolver = VehicleConflictResolver()
idd_raw = [
    {"model": "IDD", "class": "car", "confidence": 85.0, "xmin": 100, "ymin": 200, "xmax": 220, "ymax": 300},
    {"model": "IDD", "class": "motorcycle", "confidence": 35.0, "xmin": 102, "ymin": 202, "xmax": 218, "ymax": 298},
    # Legitimate rider on motorcycle
    {"model": "IDD", "class": "motorcycle", "confidence": 76.0, "xmin": 500, "ymin": 300, "xmax": 570, "ymax": 420},
    {"model": "IDD", "class": "person", "confidence": 71.0, "xmin": 505, "ymin": 290, "xmax": 565, "ymax": 410},
]
resolved = resolver.resolve(idd_raw)
classes_kept = sorted(d["class"] for d in resolved)
assert classes_kept == ["car", "motorcycle", "person"], f"Unexpected resolved classes: {classes_kept}"
print("[PASS] VehicleConflictResolver")

# 5. WeightedBoxesFuser (RDD4D + RDD_22 WBF)
fuser = WeightedBoxesFuser()
dmg_raw = [
    {"model": "RDD4D", "class": "Pot Hole", "confidence": 52.0, "xmin": 400, "ymin": 520, "xmax": 510, "ymax": 600},
    {"model": "RDD_22", "class": "pothole", "confidence": 64.0, "xmin": 405, "ymin": 518, "xmax": 515, "ymax": 605},
]
fused = fuser.fuse(dmg_raw)
assert len(fused) == 1, f"Expected 1 fused box, got {len(fused)}"
assert fused[0]["class"] == "pothole"
assert fused[0]["confidence"] > 64.0, "Cross-model agreement should boost confidence"
print(f"[PASS] WeightedBoxesFuser (boosted 64.0% -> {fused[0]['confidence']}%)")

# 6. CrossDomainSpatialFilter (Vehicles vs Traffic Signs & Sky vs Road Damage)
sfilter = CrossDomainSpatialFilter()
vehicles = [
    {"model": "IDD", "class": "truck", "confidence": 85.0, "xmin": 10, "ymin": 180, "xmax": 400, "ymax": 700},
    {"model": "IDD", "class": "bus", "confidence": 92.0, "xmin": 500, "ymin": 280, "xmax": 850, "ymax": 650},
]
signs = [
    # False positive sign sitting on the truck body
    {"model": "Traffic Sign", "class": "warning", "confidence": 52.0, "xmin": 180, "ymin": 250, "xmax": 270, "ymax": 360},
    # False positive sign wrapping the bus
    {"model": "Traffic Sign", "class": "informatory", "confidence": 48.0, "xmin": 490, "ymin": 270, "xmax": 860, "ymax": 660},
    # Legitimate roadside traffic sign outside all vehicles
    {"model": "Traffic Sign", "class": "regulatory", "confidence": 74.0, "xmin": 950, "ymin": 220, "xmax": 1020, "ymax": 295},
]
kept_signs, sup_signs = sfilter.filter_traffic_signs(signs, vehicles, 1280, 720)
assert len(kept_signs) == 1 and kept_signs[0]["class"] == "regulatory", f"Unexpected kept signs: {kept_signs}"
assert sup_signs == 2

damages = [
    # False crack in the sky
    {"model": "RDD_22", "class": "longitudinal crack", "confidence": 55.0, "xmin": 600, "ymin": 40, "xmax": 700, "ymax": 110},
    # False crack on top of the truck
    {"model": "RDD4D", "class": "alligator crack", "confidence": 48.0, "xmin": 100, "ymin": 400, "xmax": 250, "ymax": 500},
    # Valid road crack on pavement
    {"model": "RDD_22", "class": "pothole", "confidence": 66.0, "xmin": 900, "ymin": 540, "xmax": 1050, "ymax": 640},
]
kept_dmg, sup_dmg = sfilter.filter_road_damage(damages, vehicles, 1280, 720)
assert len(kept_dmg) == 1 and kept_dmg[0]["class"] == "pothole"
assert sup_dmg == 2
print("[PASS] CrossDomainSpatialFilter")

print("=" * 65)
print("ALL PIPELINE ENHANCEMENT MODULE TESTS PASSED!")
print("=" * 65)
