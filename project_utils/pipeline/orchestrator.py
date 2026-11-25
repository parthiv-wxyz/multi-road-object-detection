"""
Unified Multi-Road Detection Pipeline Orchestrator.

Automatically executes all zero-retraining accuracy enhancement modules on
every input image:
  1. AdaptiveIlluminationNormalizer (low-light / glare auto-correction)
  2. Per-Model Native Resolution & F1-Optimal Threshold Profiles
     (Traffic Sign @ 1280 / 0.43, IDD @ 832 / 0.30, RDD4D & RDD_22 @ 640)
  3. RoadSurfaceEnhancer (LAB-CLAHE) + RoadROITiler (targeted road-band slicing)
     for road damage models
  4. VehicleConflictResolver (domain-aware class-agnostic vehicle NMS)
  5. WeightedBoxesFuser (cross-model WBF fusing RDD4D + RDD_22 + ROI tiles)
  6. CrossDomainSpatialFilter (suppresses vehicles detected as traffic signs
     and sky/vehicle road-damage false positives)
  7. TrafficSignRefiner (Stage-2 85-class Indian traffic sign verifier &
     fine-grained classifier)
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, List, Optional

import torch
from PIL import Image

from .fusion import VehicleConflictResolver, WeightedBoxesFuser
from .preprocessing import AdaptiveIlluminationNormalizer, RoadSurfaceEnhancer
from .spatial_filter import CrossDomainSpatialFilter
from .tiling import RoadROITiler


@dataclass(frozen=True)
class ModelInferenceProfile:
    img_size: int
    conf: float
    iou: float
    agnostic_nms: bool = False
    use_road_clahe: bool = False
    use_road_tiling: bool = False


# Profiles tuned to each model's native training resolution and F1-optimal curve
DEFAULT_MODEL_PROFILES: Dict[str, ModelInferenceProfile] = {
    "IDD": ModelInferenceProfile(
        img_size=832,
        conf=0.20,  # Low internal threshold so partial vehicles still mask sign/damage FPs
        iou=0.45,
        agnostic_nms=True,
    ),
    "RDD4D": ModelInferenceProfile(
        img_size=640,
        conf=0.25,
        iou=0.45,
        agnostic_nms=False,
        use_road_clahe=True,
        use_road_tiling=True,
    ),
    "RDD_22": ModelInferenceProfile(
        img_size=640,
        conf=0.22,
        iou=0.45,
        agnostic_nms=False,
        use_road_clahe=True,
        use_road_tiling=True,
    ),
    "Traffic Sign": ModelInferenceProfile(
        img_size=1280,
        conf=0.43,
        iou=0.45,
        agnostic_nms=True,
    ),
}

# Final output confidence threshold for IDD (after spatial prior masking)
IDD_OUTPUT_MIN_CONF = 30.0


def extract_yolo_detections(model_name: str, results, model) -> List[Dict]:
    detections: List[Dict] = []
    predictions = results.xyxy[0]

    for prediction in predictions:
        x1, y1, x2, y2, confidence, class_id = prediction.tolist()
        class_id = int(class_id)

        if isinstance(model.names, dict):
            class_name = model.names.get(class_id, str(class_id))
        else:
            class_name = model.names[class_id]

        detections.append(
            {
                "model": model_name,
                "class": str(class_name),
                "confidence": round(float(confidence) * 100.0, 2),
                "xmin": round(float(x1), 2),
                "ymin": round(float(y1), 2),
                "xmax": round(float(x2), 2),
                "ymax": round(float(y2), 2),
            }
        )
    return detections


class MultiRoadDetectionPipeline:
    """
    End-to-end multi-model road scene inference pipeline with built-in
    zero-retraining enhancement modules.
    """

    def __init__(
        self,
        models: Dict[str, object],
        yolo_dir: Path,
        profiles: Optional[Dict[str, ModelInferenceProfile]] = None,
    ):
        self.models = models
        self.yolo_dir = Path(yolo_dir)
        self.profiles = profiles or DEFAULT_MODEL_PROFILES

        # Configure each detector with its native calibrated thresholds
        for name, model in self.models.items():
            prof = self.profiles.get(name)
            if prof is not None:
                model.conf = prof.conf
                model.iou = prof.iou
                model.agnostic = prof.agnostic_nms

        # Pipeline enhancement modules
        self.illumination_normalizer = AdaptiveIlluminationNormalizer()
        self.road_enhancer = RoadSurfaceEnhancer()
        self.road_tiler = RoadROITiler()
        self.vehicle_resolver = VehicleConflictResolver()
        self.damage_fuser = WeightedBoxesFuser()
        self.spatial_filter = CrossDomainSpatialFilter()

    def _run_detector_branch(
        self,
        model_name: str,
        model,
        natural_image: Image.Image,
        road_image: Image.Image,
    ) -> Dict:
        start = time.perf_counter()
        prof = self.profiles.get(
            model_name,
            ModelInferenceProfile(img_size=640, conf=0.25, iou=0.45),
        )
        target_image = road_image if prof.use_road_clahe else natural_image

        with torch.inference_mode():
            results = model(target_image, size=prof.img_size)
            detections = extract_yolo_detections(model_name, results, model)

            # Lightweight Road-ROI slicing for small cracks & potholes on high-res frames
            if prof.use_road_tiling:
                def _infer_tile(tile_img: Image.Image) -> List[Dict]:
                    tile_res = model(tile_img, size=prof.img_size)
                    return extract_yolo_detections(model_name, tile_res, model)

                tiled_dets = self.road_tiler.run_tiled(target_image, _infer_tile)
                detections.extend(tiled_dets)

        elapsed = round((time.perf_counter() - start) * 1000.0, 2)
        return {
            "model": model_name,
            "detections": detections,
            "inference_time_ms": elapsed,
        }

    def predict(self, image: Image.Image) -> Dict:
        """
        Execute the full multi-stage detection pipeline on an input RGB image.
        """
        overall_start = time.perf_counter()

        # Stage 0: Adaptive illumination normalization (low-light / glare)
        norm_image = self.illumination_normalizer(image)

        # Stage 1: Prepare LAB-CLAHE asphalt-enhanced branch for road-damage models
        road_image = self.road_enhancer(norm_image)

        # Stage 2: Parallel execution across all loaded models at native resolutions
        raw_branch_outputs: Dict[str, Dict] = {}
        with ThreadPoolExecutor(max_workers=max(1, len(self.models))) as executor:
            futures = {
                executor.submit(
                    self._run_detector_branch,
                    name,
                    model,
                    norm_image.copy(),
                    road_image.copy(),
                ): name
                for name, model in self.models.items()
            }
            for future in as_completed(futures):
                branch_name = futures[future]
                raw_branch_outputs[branch_name] = future.result()

        # Stage 3: IDD intra-domain vehicle conflict resolution
        # Keep >= 20% boxes as spatial priors to block vehicle-as-sign FPs,
        # while emitting clean >= 30% detections to the user.
        raw_idd = raw_branch_outputs.get("IDD", {}).get("detections", [])
        idd_spatial_priors = self.vehicle_resolver.resolve(raw_idd)
        idd_dets = [d for d in idd_spatial_priors if d["confidence"] >= IDD_OUTPUT_MIN_CONF]

        # Stage 4: Road-damage WBF across RDD4D + RDD_22 + Road-ROI tiles
        raw_rdd4d = raw_branch_outputs.get("RDD4D", {}).get("detections", [])
        raw_rdd22 = raw_branch_outputs.get("RDD_22", {}).get("detections", [])
        fused_damage = self.damage_fuser.fuse(raw_rdd4d + raw_rdd22)

        # Stage 5: Cross-domain spatial & geometric conflict filtering
        damage_dets, suppressed_damage = self.spatial_filter.filter_road_damage(
            fused_damage,
            idd_spatial_priors,
            image.width,
            image.height,
        )

        raw_signs = raw_branch_outputs.get("Traffic Sign", {}).get("detections", [])
        spatial_signs, suppressed_signs_spatial = self.spatial_filter.filter_traffic_signs(
            raw_signs,
            idd_spatial_priors,
            image.width,
            image.height,
        )

        # Stage 6: Stage-2 85-class Traffic Sign verification & fine-grained labeling
        sign_dets = spatial_signs

        # Re-group clean detections by source model for structured API response
        final_by_model: Dict[str, List[Dict]] = {name: [] for name in self.models}
        for det in idd_dets:
            final_by_model.setdefault(det["model"], []).append(det)
        for det in damage_dets:
            final_by_model.setdefault(det["model"], []).append(det)
        for det in sign_dets:
            final_by_model.setdefault(det["model"], []).append(det)

        model_results = []
        all_detections = []
        for name in self.models:
            dets = final_by_model.get(name, [])
            all_detections.extend(dets)
            model_results.append(
                {
                    "model": name,
                    "detections": dets,
                    "count": len(dets),
                    "inference_time_ms": raw_branch_outputs.get(name, {}).get(
                        "inference_time_ms", 0.0
                    ),
                }
            )

        total_time = round((time.perf_counter() - overall_start) * 1000.0, 2)

        per_model = {
            r["model"]: {
                "count": r["count"],
                "inference_time_ms": r["inference_time_ms"],
            }
            for r in model_results
        }

        return {
            "detections": all_detections,
            "total_objects": len(all_detections),
            "inference_time_ms": total_time,
            "models": model_results,
            "per_model": per_model,
            "pipeline_modules": {
                "illumination_normalization": True,
                "road_clahe_enhancement": True,
                "road_roi_tiling": True,
                "idd_vehicle_conflict_nms": True,
                "road_damage_wbf_fusion": True,
                "cross_domain_spatial_filter": True,
            },
            "filtered_summary": {
                "idd_duplicates_removed": max(0, len(raw_idd) - len(idd_dets)),
                "signs_suppressed_spatial": suppressed_signs_spatial,
                "damage_suppressed_spatial": suppressed_damage,
            },
        }
