"""
Percentile-Based Anchor Refinement

Generates optimized anchors for YOLOv5 using dataset statistics.

Author: Parthiv Project
"""

from pathlib import Path
import numpy as np
import yaml
from sklearn.cluster import KMeans


class RefinedAnchorGenerator:
    def __init__(self, img_size=640, n_anchors=9):
        self.img_size = img_size
        self.n_anchors = n_anchors

    def load_labels(self, labels_dir):
        """
        Load YOLO labels.

        Returns:
            ndarray (N,2)
            [[width,height], ...] in pixels
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

                    boxes.append([
                        w * self.img_size,
                        h * self.img_size
                    ])

        return np.array(boxes, dtype=np.float32)

    def compute_percentiles(self, boxes):
        """
        Compute width/height percentiles.
        """

        widths = boxes[:, 0]
        heights = boxes[:, 1]

        percentiles = [5, 10, 25, 50, 75, 90, 95]

        stats = {}

        for p in percentiles:
            stats[p] = {
                "width": float(np.percentile(widths, p)),
                "height": float(np.percentile(heights, p))
            }

        return stats

    def generate_anchors(self, boxes):
        """
        KMeans anchor generation.
        """

        kmeans = KMeans(
            n_clusters=self.n_anchors,
            random_state=42,
            n_init=20
        )

        kmeans.fit(boxes)

        anchors = kmeans.cluster_centers_

        areas = anchors[:, 0] * anchors[:, 1]

        order = np.argsort(areas)

        anchors = anchors[order]

        anchors = np.round(anchors).astype(int)

        return anchors

    def save_yaml(self, anchors, output_file):

        anchors = anchors.reshape(3, 3, 2)

        output = {
            "anchors": anchors.tolist()
        }

        with open(output_file, "w") as f:
            yaml.dump(output, f, sort_keys=False)

        print(f"\nSaved anchors to: {output_file}")

    def run(self, labels_dir, output_yaml):

        print("=" * 60)
        print("Percentile-Based Anchor Refinement")
        print("=" * 60)

        boxes = self.load_labels(labels_dir)

        print(f"\nLoaded {len(boxes)} bounding boxes")

        stats = self.compute_percentiles(boxes)

        print("\nPercentiles")

        for p in stats:

            print(
                f"{p:>3}%"
                f"  W={stats[p]['width']:.2f}"
                f"  H={stats[p]['height']:.2f}"
            )

        anchors = self.generate_anchors(boxes)

        print("\nGenerated Anchors")

        for a in anchors:
            print(a.tolist())

        self.save_yaml(anchors, output_yaml)

        return anchors, stats


if __name__ == "__main__":

    generator = RefinedAnchorGenerator()

    generator.run(
        labels_dir="datasets/idd/train/labels",
        output_yaml="datasets/idd/anchors.yaml"
    )