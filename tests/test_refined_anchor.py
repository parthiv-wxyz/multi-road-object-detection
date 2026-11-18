import tempfile
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_utils.refined_anchor import RefinedAnchorGenerator

print("=" * 60)
print("Refined Anchor Generator Unit Test")
print("=" * 60)

generator = RefinedAnchorGenerator(img_size=640, n_anchors=9)

# If local dataset exists, test on dataset; otherwise generate temporary labels
dataset_labels = ROOT / "datasets" / "idd" / "train" / "labels"

if dataset_labels.exists() and any(dataset_labels.glob("*.txt")):
    target_dir = dataset_labels
    yaml_out = ROOT / "datasets" / "idd" / "anchors.yaml"
    anchors, stats = generator.run(labels_dir=target_dir, output_yaml=yaml_out)
else:
    print("Local dataset not found. Testing with synthetic label data...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # Create 10 synthetic label files with various aspect ratios
        for i in range(10):
            lbl_file = tmp_path / f"sample_{i}.txt"
            with open(lbl_file, "w") as f:
                # Class x_center y_center width height
                f.write(f"0 0.5 0.5 {0.05 + i * 0.04:.4f} {0.08 + i * 0.05:.4f}\n")
                f.write(f"1 0.2 0.3 {0.02 + i * 0.01:.4f} {0.03 + i * 0.02:.4f}\n")
        
        yaml_out = tmp_path / "anchors.yaml"
        anchors, stats = generator.run(labels_dir=tmp_path, output_yaml=yaml_out)
        assert yaml_out.exists(), "YAML output file was not generated"

print(f"\nGenerated {len(anchors)} anchors:")
print(anchors)
assert len(anchors) == 9, f"Expected 9 anchors, got {len(anchors)}"
print("Refined Anchor test passed.")
