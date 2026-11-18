"""
Box fusion and intra-domain conflict resolution modules.

1. VehicleConflictResolver:
   Resolves overlapping duplicate class predictions within IDD (e.g., the same
   vehicle predicted as both 'car' and 'motorcycle', or 'autorickshaw' and
   'bus') while preserving legitimate multi-object overlaps such as a 'person'
   riding a 'motorcycle' or 'bicycle'.

2. WeightedBoxesFuser (WBF):
   Fuses overlapping road-damage predictions across RDD4D and RDD_22 (including
   full-frame and Road-ROI tiled passes) by canonicalizing shared damage
   taxonomies, computing confidence-weighted bounding boxes, and boosting
   confidence when independent models corroborate the same defect.
"""

from typing import Dict, List, Set, Tuple
import numpy as np


VEHICLE_CLASSES: Set[str] = {
    "car",
    "autorickshaw",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
}

ROAD_DAMAGE_CANONICAL_MAP: Dict[str, str] = {
    # RDD4D labels
    "alligator": "alligator crack",
    "block": "block crack",
    "longitudinal": "longitudinal crack",
    "transversal": "transverse crack",
    "pot hole": "pothole",
    # RDD_22 labels
    "alligator crack": "alligator crack",
    "longitudinal crack": "longitudinal crack",
    "transverse crack": "transverse crack",
    "pothole": "pothole",
    "other corruption": "road surface damage",
}


def compute_iou(box_a: Dict, box_b: Dict) -> float:
    """Compute IoU between two detection dictionaries."""
    ix1 = max(box_a["xmin"], box_b["xmin"])
    iy1 = max(box_a["ymin"], box_b["ymin"])
    ix2 = min(box_a["xmax"], box_b["xmax"])
    iy2 = min(box_a["ymax"], box_b["ymax"])

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, box_a["xmax"] - box_a["xmin"]) * max(
        0.0, box_a["ymax"] - box_a["ymin"]
    )
    area_b = max(0.0, box_b["xmax"] - box_b["xmin"]) * max(
        0.0, box_b["ymax"] - box_b["ymin"]
    )
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def compute_containment(inner: Dict, outer: Dict) -> float:
    """Fraction of `inner` box area covered by `outer` box."""
    ix1 = max(inner["xmin"], outer["xmin"])
    iy1 = max(inner["ymin"], outer["ymin"])
    ix2 = min(inner["xmax"], outer["xmax"])
    iy2 = min(inner["ymax"], outer["ymax"])

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    inner_area = max(0.0, inner["xmax"] - inner["xmin"]) * max(
        0.0, inner["ymax"] - inner["ymin"]
    )
    return float(inter / inner_area) if inner_area > 0.0 else 0.0


class VehicleConflictResolver:
    """
    Domain-aware class-agnostic NMS for IDD road objects.

    Suppresses duplicate vehicle class assignments on the same physical vehicle
    (e.g., car vs motorcycle or autorickshaw vs bus) while allowing 'person' to
    coexist with 'motorcycle' / 'bicycle' (two-wheeler riders).
    """

    def __init__(self, iou_threshold: float = 0.52, containment_threshold: float = 0.85):
        self.iou_threshold = iou_threshold
        self.containment_threshold = containment_threshold

    def resolve(self, detections: List[Dict]) -> List[Dict]:
        if len(detections) <= 1:
            return list(detections)

        sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        kept: List[Dict] = []

        for cand in sorted_dets:
            cand_cls = cand["class"].lower()
            suppressed = False

            for prev in kept:
                prev_cls = prev["class"].lower()

                # Allow person riding motorcycle/bicycle
                if {cand_cls, prev_cls} in (
                    {"person", "motorcycle"},
                    {"person", "bicycle"},
                ):
                    continue

                iou = compute_iou(cand, prev)
                if iou >= self.iou_threshold:
                    # Same class or two mutually exclusive vehicle classes
                    if cand_cls == prev_cls or (
                        cand_cls in VEHICLE_CLASSES and prev_cls in VEHICLE_CLASSES
                    ):
                        suppressed = True
                        break

                # Suppress nearly identical duplicate sub-boxes of same class
                if cand_cls == prev_cls:
                    if compute_containment(cand, prev) >= self.containment_threshold:
                        suppressed = True
                        break

            if not suppressed:
                kept.append(cand)

        return kept


class WeightedBoxesFuser:
    """
    Cross-model Weighted Boxes Fusion (WBF) for road-damage detectors
    (RDD4D + RDD_22 across full-frame and Road-ROI tile passes).
    """

    def __init__(
        self,
        iou_threshold: float = 0.45,
        min_fused_conf: float = 28.0,
        model_weights: Dict[str, float] = None,
    ):
        self.iou_threshold = iou_threshold
        self.min_fused_conf = min_fused_conf
        # RDD_22 has higher validation mAP@0.5 (57.4%) vs RDD4D (39.3%)
        self.model_weights = model_weights or {
            "RDD_22": 1.25,
            "RDD4D": 0.90,
        }

    @staticmethod
    def canonicalize_class(class_name: str) -> str:
        return ROAD_DAMAGE_CANONICAL_MAP.get(class_name.strip().lower(), class_name)

    def fuse(self, detections: List[Dict]) -> List[Dict]:
        if not detections:
            return []

        # Normalize class names and sort descending by confidence
        normalized = []
        for det in detections:
            item = dict(det)
            item["canonical_class"] = self.canonicalize_class(det["class"])
            normalized.append(item)

        normalized.sort(key=lambda d: d["confidence"], reverse=True)

        # Group into overlapping clusters (class-agnostic for overlapping damage on same spot)
        clusters: List[List[Dict]] = []
        cluster_boxes: List[Dict] = []

        for det in normalized:
            best_idx = -1
            best_iou = 0.0

            for idx, rep_box in enumerate(cluster_boxes):
                iou = compute_iou(det, rep_box)
                if iou >= self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx >= 0:
                clusters[best_idx].append(det)
                # Recompute weighted representative box
                cluster_boxes[best_idx] = self._weighted_box(clusters[best_idx])
            else:
                clusters.append([det])
                cluster_boxes.append(self._weighted_box([det]))

        fused_results: List[Dict] = []
        for cluster, rep in zip(clusters, cluster_boxes):
            distinct_models = {d["model"] for d in cluster}
            max_conf = max(d["confidence"] for d in cluster)

            # Ensemble corroboration boost if both RDD4D and RDD_22 agree
            if len(distinct_models) >= 2:
                avg_conf = float(np.mean([d["confidence"] for d in cluster]))
                fused_conf = min(99.5, max_conf + 0.20 * avg_conf)
                source_model = "RDD_22" if "RDD_22" in distinct_models else cluster[0]["model"]
            else:
                # Single-model detection: mild boost if seen in both full-frame & tile
                if len(cluster) >= 2:
                    fused_conf = min(99.0, max_conf * 1.08)
                else:
                    fused_conf = max_conf
                source_model = cluster[0]["model"]

            if fused_conf < self.min_fused_conf:
                continue

            # Pick class from the highest weighted vote
            class_scores: Dict[str, float] = {}
            for d in cluster:
                w = self.model_weights.get(d["model"], 1.0) * d["confidence"]
                c_name = d["canonical_class"]
                class_scores[c_name] = class_scores.get(c_name, 0.0) + w
            best_class = max(class_scores.items(), key=lambda kv: kv[1])[0]

            fused_results.append(
                {
                    "model": source_model,
                    "class": best_class,
                    "confidence": round(float(fused_conf), 2),
                    "xmin": round(float(rep["xmin"]), 2),
                    "ymin": round(float(rep["ymin"]), 2),
                    "xmax": round(float(rep["xmax"]), 2),
                    "ymax": round(float(rep["ymax"]), 2),
                }
            )

        fused_results.sort(key=lambda d: d["confidence"], reverse=True)
        return fused_results

    def _weighted_box(self, cluster: List[Dict]) -> Dict:
        weights = np.array(
            [
                max(1e-3, d["confidence"]) * self.model_weights.get(d["model"], 1.0)
                for d in cluster
            ],
            dtype=np.float64,
        )
        w_sum = float(np.sum(weights))
        return {
            "xmin": float(np.sum([d["xmin"] * w for d, w in zip(cluster, weights)]) / w_sum),
            "ymin": float(np.sum([d["ymin"] * w for d, w in zip(cluster, weights)]) / w_sum),
            "xmax": float(np.sum([d["xmax"] * w for d, w in zip(cluster, weights)]) / w_sum),
            "ymax": float(np.sum([d["ymax"] * w for d, w in zip(cluster, weights)]) / w_sum),
        }
