"""
Targeted Road-Surface ROI Tiling (Lightweight SAHI) for small crack and
pothole detection without full-image tiling overhead.

On high-resolution dashcam/road images, road damage resides primarily in the
lower half of the frame and loses fine pixel detail when resized to 640x640.
RoadROITiler extracts overlapping tiles across the lower road band, runs the
detector on each tile, and projects boxes back into full-image coordinates.
"""

from typing import Callable, Dict, List, Tuple
from PIL import Image


class RoadROITiler:
    """
    Generates overlapping road-surface crops from the lower band of high-res
    images and maps detections back to original image coordinates.
    """

    def __init__(
        self,
        min_width: int = 960,
        min_height: int = 540,
        road_start_ratio: float = 0.42,
        overlap_ratio: float = 0.18,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.road_start_ratio = road_start_ratio
        self.overlap_ratio = overlap_ratio

    def generate_tiles(self, image: Image.Image) -> List[Tuple[Image.Image, int, int]]:
        """
        Return list of (tile_image, x_offset, y_offset) tuples covering the
        lower road-surface band. Returns empty list for small images where
        full-frame inference already preserves sufficient pixel density.
        """
        w, h = image.size
        if w < self.min_width or h < self.min_height:
            return []

        y0 = int(h * self.road_start_ratio)
        y1 = h

        # Two overlapping horizontal tiles covering the road band
        mid = w // 2
        pad = int(w * (self.overlap_ratio / 2.0))

        left_box = (0, y0, min(w, mid + pad), y1)
        right_box = (max(0, mid - pad), y0, w, y1)

        return [
            (image.crop(left_box), left_box[0], left_box[1]),
            (image.crop(right_box), right_box[0], right_box[1]),
        ]

    def project_detections(
        self,
        tile_detections: List[Dict],
        x_offset: int,
        y_offset: int,
        full_width: int,
        full_height: int,
    ) -> List[Dict]:
        """
        Translate tile-local bounding boxes back to full-image coordinates.
        """
        projected = []
        for det in tile_detections:
            xmin = max(0.0, min(float(full_width), det["xmin"] + x_offset))
            ymin = max(0.0, min(float(full_height), det["ymin"] + y_offset))
            xmax = max(0.0, min(float(full_width), det["xmax"] + x_offset))
            ymax = max(0.0, min(float(full_height), det["ymax"] + y_offset))

            if xmax <= xmin + 2.0 or ymax <= ymin + 2.0:
                continue

            item = dict(det)
            item["xmin"] = round(xmin, 2)
            item["ymin"] = round(ymin, 2)
            item["xmax"] = round(xmax, 2)
            item["ymax"] = round(ymax, 2)
            projected.append(item)
        return projected

    def run_tiled(
        self,
        image: Image.Image,
        infer_fn: Callable[[Image.Image], List[Dict]],
    ) -> List[Dict]:
        """
        Run inference on road-surface ROI tiles and project boxes back.
        """
        tiles = self.generate_tiles(image)
        if not tiles:
            return []

        full_w, full_h = image.size
        all_projected = []
        for tile_img, x_off, y_off in tiles:
            dets = infer_fn(tile_img)
            all_projected.extend(
                self.project_detections(dets, x_off, y_off, full_w, full_h)
            )
        return all_projected
