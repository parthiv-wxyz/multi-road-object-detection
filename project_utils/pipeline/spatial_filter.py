"""
Cross-Domain Spatial & Geometric Conflict Filter.

Uses detections from the IDD road-scene model (vehicles & pedestrians) and
scene geometry priors to suppress physically impossible cross-model false
positives without requiring model retraining:

1. Eliminates vehicles, windshields, rear boards, headlights, and decals being
   misclassified as Traffic Signs.
2. Eliminates road damage (cracks/potholes) predicted on top of vehicles or in
   the upper sky/horizon zone.
3. Enforces traffic sign geometric sanity (minimum size, aspect ratio, and
   position constraints) while preserving close-up sign photos.
"""

from typing import Dict, List, Tuple
from .fusion import compute_containment, compute_iou


class CrossDomainSpatialFilter:
    """
    Filters Traffic Sign and Road Damage candidates using IDD vehicle/person
    bounding boxes and scene geometry priors.
    """

    def __init__(
        self,
        sign_in_vehicle_thresh: float = 0.35,
        vehicle_in_sign_thresh: float = 0.45,
        sign_vehicle_iou_thresh: float = 0.25,
        damage_in_vehicle_thresh: float = 0.40,
        damage_horizon_ratio: float = 0.30,
    ):
        self.sign_in_vehicle_thresh = sign_in_vehicle_thresh
        self.vehicle_in_sign_thresh = vehicle_in_sign_thresh
        self.sign_vehicle_iou_thresh = sign_vehicle_iou_thresh
        self.damage_in_vehicle_thresh = damage_in_vehicle_thresh
        self.damage_horizon_ratio = damage_horizon_ratio

    @staticmethod
    def _box_metrics(det: Dict) -> Tuple[float, float, float, float, float]:
        w = max(0.0, det["xmax"] - det["xmin"])
        h = max(0.0, det["ymax"] - det["ymin"])
        area = w * h
        cx = 0.5 * (det["xmin"] + det["xmax"])
        cy = 0.5 * (det["ymin"] + det["ymax"])
        return w, h, area, cx, cy

    def filter_traffic_signs(
        self,
        sign_detections: List[Dict],
        idd_detections: List[Dict],
        image_width: int,
        image_height: int,
    ) -> Tuple[List[Dict], int]:
        """
        Remove false-positive Traffic Sign predictions that overlap vehicles
        or violate geometric sanity rules.

        Returns:
            (kept_signs, suppressed_count)
        """
        img_area = float(max(1, image_width * image_height))
        has_vehicles = len(idd_detections) > 0
        kept: List[Dict] = []
        suppressed = 0

        for sign in sign_detections:
            sw, sh, s_area, _, s_cy = self._box_metrics(sign)
            conf = sign["confidence"]

            # 1. Minimum pixel dimensions
            if sw < 10.0 or sh < 10.0:
                suppressed += 1
                continue

            # 2. Aspect ratio sanity (allows circular/triangular/rectangular boards,
            #    rejects narrow horizontal vehicle trim or long thin poles)
            aspect = sw / max(1e-5, sh)
            if aspect < 0.28 or aspect > 3.5:
                suppressed += 1
                continue

            # 3. Area sanity (stricter in multi-vehicle driving scenes; relaxed for
            #    dedicated close-up sign images with no vehicles present)
            max_area_ratio = 0.16 if has_vehicles else 0.65
            if (s_area / img_area) > max_area_ratio and conf < 80.0:
                suppressed += 1
                continue

            # 4. Bottom road-surface rejection (signs are mounted above pavement)
            if s_cy > 0.88 * image_height and conf < 75.0:
                suppressed += 1
                continue

            # 5. Cross-model Vehicle / Person conflict suppression
            vehicle_conflict = False
            for obj in idd_detections:
                vw, vh, v_area, _, _ = self._box_metrics(obj)
                if v_area <= 0.0:
                    continue

                in_veh = compute_containment(sign, obj)
                veh_in_sign = compute_containment(obj, sign)
                iou = compute_iou(sign, obj)

                if (
                    in_veh >= self.sign_in_vehicle_thresh
                    or veh_in_sign >= self.vehicle_in_sign_thresh
                    or iou >= self.sign_vehicle_iou_thresh
                ):
                    # Narrow exception: genuinely high-confidence small pole sign
                    # peeking just above a vehicle roof line
                    is_roof_edge_sign = (
                        conf >= 82.0
                        and s_area < 0.08 * v_area
                        and sign["ymax"] <= obj["ymin"] + 0.20 * vh
                    )
                    if not is_roof_edge_sign:
                        vehicle_conflict = True
                        break

            if vehicle_conflict:
                suppressed += 1
                continue

            kept.append(sign)

        return kept, suppressed

    def filter_road_damage(
        self,
        damage_detections: List[Dict],
        idd_detections: List[Dict],
        image_width: int,
        image_height: int,
    ) -> Tuple[List[Dict], int]:
        """
        Remove false-positive road damage predictions in the sky/horizon zone
        or on top of vehicles/pedestrians.

        Returns:
            (kept_damage, suppressed_count)
        """
        kept: List[Dict] = []
        suppressed = 0

        for dmg in damage_detections:
            dw, dh, d_area, _, d_cy = self._box_metrics(dmg)

            # Reject tiny 1-pixel speckles
            if dw < 8.0 or dh < 8.0:
                suppressed += 1
                continue

            # Reject upper sky/trees/buildings zone
            if d_cy < self.damage_horizon_ratio * image_height:
                suppressed += 1
                continue

            # Reject cracks/potholes projected onto vehicles or pedestrians
            on_vehicle = False
            for obj in idd_detections:
                if (
                    compute_containment(dmg, obj) >= self.damage_in_vehicle_thresh
                    or compute_iou(dmg, obj) >= 0.35
                ):
                    on_vehicle = True
                    break

            if on_vehicle:
                suppressed += 1
                continue

            kept.append(dmg)

        return kept, suppressed
