import cv2
import torch
from pathlib import Path

from utils.augmentations import classify_transforms
from utils.torch_utils import select_device


CLASSIFIER_WEIGHTS = r"runs\train-cls\traffic_sign_cls_50\weights\best.pt"

CROP_DIR = r"runs\traffic_sign_pipeline_v2_crops"

IMG_SIZE = 224
DEVICE = "cpu"

device = select_device(DEVICE)

# Load classifier
checkpoint = torch.load(
    CLASSIFIER_WEIGHTS,
    map_location=device,
    weights_only=False
)

model = checkpoint["model"].float().eval()
names = model.names

transform = classify_transforms(IMG_SIZE)

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG"
}

crop_dir = Path(CROP_DIR)

images = [
    p for p in crop_dir.iterdir()
    if p.suffix in extensions
]

print("\n========================================")
print(" DIRECT CLASSIFIER CROP TEST")
print("========================================")

for image_path in images:

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    tensor = transform(image)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        top_values, top_indices = torch.topk(
            probabilities,
            5,
            dim=1
        )

    print("\n" + image_path.name)

    for rank in range(5):

        class_id = int(
            top_indices[0][rank]
        )

        confidence = float(
            top_values[0][rank]
        )

        print(
            f"  {rank + 1}. "
            f"{names[class_id]:45s} "
            f"{confidence:.3f}"
        )

print("\n========================================")