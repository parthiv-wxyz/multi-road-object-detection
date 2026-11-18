"""
Percentile-Based Anchor Refinement for YOLOv5

Generates optimized anchors for YOLOv5 architectures using dataset statistics
and K-Means clustering.
"""

from pathlib import Path
import numpy as np
import yaml
from sklearn.cluster import KMeans


class RefinedAnchorGenerator:
    """
    K-Means and percentile-based anchor generator for YOLO models.
    """

    def __init__(self, img_size=640, n_anchors=9):
        self.img_size = img_size
        self.n_anchors = n_anchors

    def load_labels(self, labels_dir):
        """
        Load YOLO labels from text files.

        Returns:
            ndarray (N, 2): [[width, height], ...] in pixel units
        """
        labels_dir = Path(labels_dir)
        boxes = []
        files = list(labels_dir.rglob("*.txt"))

        if len(files) == 0:
            raise FileNotFoundError(f"No label files found in {labels_dir}")

        for file in files:
            with open(file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    _, _, _, w, h = map(float, parts)
                    boxes.append([w * self.img_size, h * self.img_size])

        return np.array(boxes, dtype=np.float32)

    def compute_percentiles(self, boxes):
        """
        Compute width/height percentiles across bounding boxes.
        """
        widths = boxes[:, 0]
        heights = boxes[:, 1]
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        stats = {}

        for p in percentiles:
            stats[p] = {
                "width": float(np.percentile(widths, p)),
                "height": float(np.percentile(heights, p)),
            }

        return stats

    def generate_anchors(self, boxes):
        """
        K-Means anchor generation.
        """
        kmeans = KMeans(
            n_clusters=self.n_anchors,
            random_state=42,
            n_init=20,
        )
        kmeans.fit(boxes)
        anchors = kmeans.cluster_centers_

        # Sort anchors by bounding box area
        areas = anchors[:, 0] * anchors[:, 1]
        order = np.argsort(areas)
        anchors = anchors[order]

        return np.round(anchors).astype(int)

    def save_yaml(self, anchors, output_file):
        """
        Save anchors formatted for YOLOv5 model configuration YAML.
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        n_groups = 4 if len(anchors) == 12 else 3
        anchors_per_group = len(anchors) // n_groups
        reshaped = anchors.reshape(n_groups, anchors_per_group, 2)

        output = {"anchors": reshaped.tolist()}
        with open(output_file, "w") as f:
            yaml.dump(output, f, sort_keys=False)

        print(f"Saved anchors to: {output_file}")

    def run(self, labels_dir, output_yaml=None):
        """
        Run anchor generation pipeline.
        """
        boxes = self.load_labels(labels_dir)
        stats = self.compute_percentiles(boxes)
        anchors = self.generate_anchors(boxes)

        if output_yaml:
            self.save_yaml(anchors, output_yaml)

        return anchors, stats
