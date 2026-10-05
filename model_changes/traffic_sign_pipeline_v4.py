import os
import cv2
import torch
import numpy as np

from pathlib import Path

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox, classify_transforms


# ============================================================
# CONFIGURATION
# ============================================================

DETECTOR_WEIGHTS = (
    r"runs\train\traffic_sign_enhanced_1280_pretrained"
    r"\weights\best.pt"
)

CLASSIFIER_WEIGHTS = (
    r"runs\train-cls\traffic_sign_cls_50"
    r"\weights\best.pt"
)

SOURCE = r"E:\Parthiv\multi-road-object-detection\datasets\traffic_sign\images\test"

OUTPUT_DIR = r"runs\traffic_sign_pipeline_v4_conf40_dconf35_pad0"
CROP_DIR = r"runs\traffic_sign_pipeline_v4_conf40_dconf35_pad0_crops"

DEVICE = "0"

DETECT_IMG_SIZE = 1280
CLASSIFY_IMG_SIZE = 224

# Detector threshold
DETECT_CONF = 0.40

# Detector NMS IoU
DETECT_IOU = 0.45

# Extra duplicate suppression
DUPLICATE_IOU = 0.60

# Classifier threshold
CLASSIFY_CONF = 0.35

# Padding around detector box
CROP_PADDING = 0.00


# ============================================================
# DEVICE
# ============================================================

device = select_device(DEVICE)


# ============================================================
# LOAD DETECTOR
# ============================================================

print("\nLoading detector...")

detector = attempt_load(
    DETECTOR_WEIGHTS,
    device=device
)

detector.eval()

# FP16 detector inference on RTX 4070.
if device.type != "cpu":
    detector.half()

print("Detector loaded.")


# ============================================================
# LOAD CLASSIFIER
# ============================================================

print("\nLoading classifier...")

checkpoint = torch.load(
    CLASSIFIER_WEIGHTS,
    map_location=device,
    weights_only=False
)

classifier = checkpoint["model"].float().eval()

classifier_names = classifier.names

CLASS_TO_ID = {
    name: i
    for i, name in enumerate(classifier_names)
}

print("Classifier loaded.")
print(f"Classifier classes: {len(classifier_names)}")


# ============================================================
# YOLOV5 CLASSIFICATION PREPROCESSING
# ============================================================

classification_transform = classify_transforms(
    CLASSIFY_IMG_SIZE
)


# ============================================================
# IOU FUNCTION
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection = (
        intersection_width *
        intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# REMOVE DUPLICATE DETECTIONS
# ============================================================

def remove_duplicate_detections(detections):

    if len(detections) <= 1:
        return detections

    # Sort by detector confidence
    detections = sorted(
        detections,
        key=lambda x: x[4],
        reverse=True
    )

    kept = []

    for detection in detections:

        box = detection[:4]

        duplicate = False

        for existing in kept:

            existing_box = existing[:4]

            iou = calculate_iou(
                box,
                existing_box
            )

            if iou >= DUPLICATE_IOU:

                duplicate = True
                break

        if not duplicate:
            kept.append(detection)

    return kept


# ============================================================
# CROP WITH PADDING
# ============================================================

def crop_with_padding(image, box):

    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    pad_x = int(width * CROP_PADDING)
    pad_y = int(height * CROP_PADDING)

    x1 = max(
        0,
        x1 - pad_x
    )

    y1 = max(
        0,
        y1 - pad_y
    )

    x2 = min(
        image.shape[1],
        x2 + pad_x
    )

    y2 = min(
        image.shape[0],
        y2 + pad_y
    )

    return image[y1:y2, x1:x2]


# ============================================================
# CLASSIFY SIGN
# ============================================================

# ============================================================
# CLASSIFY SIGN
# ============================================================

def classify_sign(crop):

    crop_rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    tensor = classification_transform(
        crop_rgb
    )

    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = classifier(tensor)[0]

    probabilities = torch.softmax(
        logits,
        dim=0
    )

    top_k = min(3, len(classifier_names))

    top_probabilities, top_ids = torch.topk(
        probabilities,
        top_k
    )

    top_predictions = [
        (
            classifier_names[int(class_id)],
            float(confidence)
        )
        for confidence, class_id in zip(
            top_probabilities,
            top_ids
        )
    ]

    class_name = top_predictions[0][0]
    class_confidence = top_predictions[0][1]

    if class_confidence < CLASSIFY_CONF:
        class_name = "UNKNOWN"

    return (
        class_name,
        class_confidence,
        top_predictions
    )


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(
    image_path,
    output_path
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            f"Could not read: {image_path}"
        )

        return

    original = image.copy()

    # --------------------------------------------------------
    # Detector preprocessing
    # --------------------------------------------------------

    img = letterbox(
        image,
        DETECT_IMG_SIZE,
        stride=32,
        auto=True
    )[0]

    img = (
        img[:, :, ::-1]
        .transpose(2, 0, 1)
    )

    img = np.ascontiguousarray(
        img
    )

    img = torch.from_numpy(
        img
    ).to(device)

    if device.type != "cpu":
        img = img.half() / 255.0
    else:
        img = img.float() / 255.0

    if img.ndimension() == 3:

        img = img.unsqueeze(0)

    # --------------------------------------------------------
    # Detection
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = detector(
            img
        )[0]

    detections = non_max_suppression(
        prediction,
        DETECT_CONF,
        DETECT_IOU
    )

    all_detections = []

    for detection in detections:

        if len(detection) == 0:
            continue

        detection[:, :4] = scale_boxes(
            img.shape[2:],
            detection[:, :4],
            original.shape
        ).round()

        for row in detection:

            x1, y1, x2, y2 = map(
                int,
                row[:4]
            )

            detector_confidence = float(
                row[4].item()
            )

            detector_class_id = int(
                row[5].item()
            )

            all_detections.append([
                x1,
                y1,
                x2,
                y2,
                detector_confidence,
                detector_class_id
            ])

    # --------------------------------------------------------
    # Additional duplicate removal
    # --------------------------------------------------------

    all_detections = (
        remove_duplicate_detections(
            all_detections
        )
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    detection_count = 0

    for detection in all_detections:

        (
            x1,
            y1,
            x2,
            y2,
            detector_confidence,
            detector_class_id
        ) = detection

        # Crop with padding
        crop = crop_with_padding(
            original,
            (
                x1,
                y1,
                x2,
                y2
            )
        )

        crop_filename = (
            f"{image_path.stem}_"
            f"{detection_count + 1}.jpg"
        )

        if crop.size == 0:
            continue

        crop_path = (
            Path(CROP_DIR) /
            crop_filename
        )

        cv2.imwrite(
            str(crop_path),
            crop
        )

        if crop.size == 0:
            continue

        # ----------------------------------------------------
        # Detector broad class
        # ----------------------------------------------------

        detector_class_name = (
            detector.names[
                detector_class_id
            ]
        )

        class_name, class_confidence, top_predictions = (
            classify_sign(crop)
        )

        detection_count += 1


        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        if class_name == "UNKNOWN":

            label = (
                f"UNKNOWN "
                f"({class_confidence:.2f})"
            )

        else:

            label = (
                f"{class_name} "
                f"{class_confidence:.2f}"
            )

        # ----------------------------------------------------
        # Draw bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            original,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2
        )

        text_y = max(
            y1,
            text_height + 5
        )

        cv2.rectangle(
            original,
            (
                x1,
                text_y -
                text_height -
                baseline -
                5
            ),
            (
                x1 +
                text_width +
                5,
                text_y
            ),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            original,
            label,
            (
                x1 + 2,
                text_y - 4
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        print(
            f"  Detector: "
            f"{detector_class_name} "
            f"{detector_confidence:.3f}"
        )

        print(
            f"  Classifier: "
            f"{class_name} "
            f"{class_confidence:.3f}"
        )

        print("  Top-3:")
        for rank, (name, confidence) in enumerate(
            top_predictions,
            1
        ):
            print(
                f"    {rank}. {name} "
                f"{confidence:.3f}"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    cv2.imwrite(
        str(output_path),
        original
    )

    print(
        f"  Detections: "
        f"{detection_count}"
    )

    print(
        f"  Saved: "
        f"{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    os.makedirs(
        CROP_DIR,
        exist_ok=True
    )

    source = Path(SOURCE)

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".JPG",
        ".JPEG",
        ".PNG"
    }

    images = [
        p
        for p in source.iterdir()
        if p.suffix in image_extensions
    ]

    print()
    print("=" * 55)
    print(" Traffic Sign Detection + Classification V4")
    print("=" * 55)

    print(
        f"Images found: {len(images)}"
    )

    print(
        f"Detector confidence: "
        f"{DETECT_CONF}"
    )

    print(
        f"Classifier confidence: "
        f"{CLASSIFY_CONF}"
    )

    print(
        f"Crop padding: "
        f"{CROP_PADDING * 100:.0f}%"
    )

    print(f"Device: {device}")
    print(f"Detector image size: {DETECT_IMG_SIZE}")
    print(f"Classifier image size: {CLASSIFY_IMG_SIZE}")

    for index, image_path in enumerate(
        images,
        1
    ):

        print()
        print(
            f"[{index}/{len(images)}] "
            f"{image_path.name}"
        )

        output_path = (
            Path(OUTPUT_DIR)
            / image_path.name
        )

        process_image(
            image_path,
            output_path
        )

    print()
    print("=" * 55)
    print("Pipeline completed.")
    print(
        f"Results: {OUTPUT_DIR}"
    )
    print("=" * 55)


if __name__ == "__main__":

    main()