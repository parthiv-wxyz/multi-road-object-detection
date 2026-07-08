from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.refined_anchor import RefinedAnchorGenerator

generator = RefinedAnchorGenerator()

anchors, stats = generator.run(
    labels_dir="datasets/idd/train/labels",
    output_yaml="datasets/idd/anchors.yaml"
)

print("\nFinal Anchors")
print(anchors)