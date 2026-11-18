"""
Stage-2 Traffic Sign Verifier & Fine-Grained Classifier.

Leverages the already-trained 85-class Indian Traffic Sign classification model
(`yolov5/runs/train-cls/traffic_sign_cls_50/weights/best.pt`, 94.18% Top-1 acc)
inside the inference pipeline to:

1. Verify Stage-1 Traffic Sign detector proposals and reject background/vehicle
   false positives that do not resemble any real Indian traffic sign.
2. Upgrade generic Stage-1 categories ('regulatory', 'mandatory', 'warning',
   'informatory', 'general') into exact fine-grained Indian traffic sign names
   (e.g., 'Stop', 'Speed Limit 50', 'School Ahead', 'No Parking').
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _format_sign_label(raw_name: str) -> str:
    """Convert 'SPEED_LIMIT_50' -> 'Speed Limit 50' and fix known dataset typos."""
    cleaned = raw_name.replace("_", " ").strip()
    if cleaned.upper() == "STOP":
        return "Stop"
    formatted = " ".join(
        word.capitalize() if not word.isdigit() else word
        for word in cleaned.split()
    )
    return formatted.replace("Compulsary", "Compulsory")


class TrafficSignRefiner:
    """
    Two-stage verifier and fine-grained classifier for traffic sign crops.
    """

    def __init__(
        self,
        weights_path: Path,
        yolo_dir: Path,
        img_size: int = 224,
        crop_padding: float = 0.08,
        reject_det_conf_ceiling: float = 68.0,
        min_cls_verify_prob: float = 0.28,
        min_cls_enrich_prob: float = 0.42,
    ):
        self.weights_path = Path(weights_path)
        self.yolo_dir = Path(yolo_dir)
        self.img_size = img_size
        self.crop_padding = crop_padding
        self.reject_det_conf_ceiling = reject_det_conf_ceiling
        self.min_cls_verify_prob = min_cls_verify_prob
        self.min_cls_enrich_prob = min_cls_enrich_prob

        self.model = None
        self.names: Dict[int, str] = {}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._load_classifier()

    def _load_classifier(self) -> None:
        if not self.weights_path.exists():
            print(f"[TrafficSignRefiner] Classifier weights not found at {self.weights_path}; Stage-2 disabled.")
            return

        if str(self.yolo_dir) not in sys.path:
            sys.path.insert(0, str(self.yolo_dir))

        checkpoint = torch.load(
            str(self.weights_path),
            map_location=self.device,
            weights_only=False,
        )
        model = checkpoint["model"].float().to(self.device).eval()
        self.model = model

        if isinstance(model.names, dict):
            self.names = {int(k): str(v) for k, v in model.names.items()}
        else:
            self.names = {idx: str(name) for idx, name in enumerate(model.names)}

        print(
            f"[TrafficSignRefiner] Loaded 85-class verifier ({len(self.names)} classes) "
            f"from {self.weights_path.name}"
        )

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def _preprocess_crop(self, crop: Image.Image) -> torch.Tensor:
        """
        Resize RGB crop to (img_size, img_size) matching YOLOv5 classification
        ImageNet normalization.
        """
        resized = crop.convert("RGB").resize(
            (self.img_size, self.img_size), Image.Resampling.BILINEAR
        )
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        # HWC -> CHW -> NCHW
        tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
        return tensor.unsqueeze(0).to(self.device)

    def _extract_crop(self, image: Image.Image, det: Dict) -> Optional[Image.Image]:
        w, h = image.size
        bw = det["xmax"] - det["xmin"]
        bh = det["ymax"] - det["ymin"]
        if bw < 6 or bh < 6:
            return None

        pad_x = bw * self.crop_padding
        pad_y = bh * self.crop_padding

        x1 = max(0, int(det["xmin"] - pad_x))
        y1 = max(0, int(det["ymin"] - pad_y))
        x2 = min(w, int(det["xmax"] + pad_x))
        y2 = min(h, int(det["ymax"] + pad_y))

        if x2 <= x1 + 4 or y2 <= y1 + 4:
            return None
        return image.crop((x1, y1, x2, y2))

    def refine(
        self,
        image: Image.Image,
        sign_detections: List[Dict],
    ) -> Tuple[List[Dict], int]:
        """
        Verify and refine traffic sign detections.

        Returns:
            (refined_detections, rejected_count)
        """
        if not sign_detections or not self.is_ready:
            return list(sign_detections), 0

        kept: List[Dict] = []
        rejected = 0

        with torch.inference_mode():
            for det in sign_detections:
                crop = self._extract_crop(image, det)
                if crop is None:
                    rejected += 1
                    continue

                tensor = self._preprocess_crop(crop)
                logits = self.model(tensor)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                probs = F.softmax(logits, dim=1)[0]
                top1_prob, top1_idx = torch.max(probs, dim=0)
                prob_val = float(top1_prob.item())
                cls_id = int(top1_idx.item())
                raw_sign_name = self.names.get(cls_id, "")

                det_conf = det["confidence"]
                det_cat = det["class"].strip().lower()

                # Category-aware verification gate:
                # - 'informatory' / 'general' had the weakest validation AP (0.50) and
                #   are prone to firing on truck boards/billboards; require higher
                #   Stage-1 confidence unless corroborated by Stage-2 classifier.
                # - 'regulatory' / 'mandatory' / 'warning' belong to the 85-class
                #   Indian sign taxonomy; reject uncertain Stage-1 proposals that fail
                #   Stage-2 verification.
                if det_cat in {"informatory", "general"}:
                    if det_conf < 52.0 and prob_val < 0.35:
                        rejected += 1
                        continue
                    enrich_thresh = 0.55
                else:
                    if (
                        det_conf < self.reject_det_conf_ceiling
                        and prob_val < self.min_cls_verify_prob
                    ):
                        rejected += 1
                        continue
                    enrich_thresh = self.min_cls_enrich_prob

                item = dict(det)
                item["category"] = det["class"]
                item["classifier_confidence"] = round(prob_val * 100.0, 2)

                # Enrich generic label with exact Indian sign name when confident
                if prob_val >= enrich_thresh and raw_sign_name:
                    pretty = _format_sign_label(raw_sign_name)
                    item["class"] = f"{pretty} ({det['class']})"

                kept.append(item)

        return kept, rejected
