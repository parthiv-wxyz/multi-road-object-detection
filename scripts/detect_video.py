"""
detect_video.py

Inference script for processing video streams with YOLO models.
Renders bounding boxes and saves the processed output video.

Usage:
    python scripts/detect_video.py --weights yolov5/runs/train/idd_final/weights/best.pt --source test_video.mp4 --output output.mp4
"""

import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO object detection on video")
    parser.add_argument("--weights", type=str, default="yolov5/runs/train/idd_final/weights/best.pt", help="Path to weights")
    parser.add_argument("--source", type=str, required=True, help="Video source file or RTSP/camera index")
    parser.add_argument("--output", type=str, default="runs/detect/video_out.mp4", help="Output video path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--img-size", type=int, default=640, help="Inference resolution")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' or '0'")
    return parser.parse_args()

def main():
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"Loading model from {weights_path} on {device}...")
    model = torch.hub.load("yolov5", "custom", path=str(weights_path), source="local")
    model.conf = args.conf
    model.iou = args.iou
    model.to(device)
    model.eval()

    # Open video
    source_is_cam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if source_is_cam else args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not source_is_cam else -1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    print(f"Processing video: {width}x{height} @ {fps:.1f} FPS, {total_frames} total frames")
    frame_idx = 0
    start_time = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Inference expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            results = model(rgb_frame, size=args.img_size)

        # Draw detections
        det = results.xyxy[0].cpu().numpy()
        for x1, y1, x2, y2, conf, cls_id in det:
            cls_name = model.names[int(cls_id)]
            label = f"{cls_name} {conf:.2f}"
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(frame, p1, p2, (0, 220, 100), 2)
            cv2.putText(frame, label, (p1[0], max(0, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 100), 2)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0:
            elapsed = time.perf_counter() - start_time
            print(f"  Frame {frame_idx}/{total_frames} ({frame_idx / elapsed:.1f} FPS)")

    cap.release()
    writer.release()
    print(f"Video saved to {out_path} ({frame_idx} frames processed)")

if __name__ == "__main__":
    main()
