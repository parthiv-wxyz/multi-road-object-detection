"""
benchmark.py

Benchmarking utility for YOLOv5 models.
Measures latency (mean, std, p50, p95), FPS, model parameters, and GFLOPs.

Usage:
    python scripts/benchmark.py --weights yolov5/runs/train/idd_final/weights/best.pt --img 640 --device cpu
"""

import argparse
import time
from pathlib import Path
import numpy as np
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark YOLO model inference performance")
    parser.add_argument("--weights", type=str, default="yolov5s.pt", help="Path to model weights (.pt)")
    parser.add_argument("--img", type=int, default=640, help="Inference image resolution (square)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for benchmarking")
    parser.add_argument("--warmup", type=int, default=15, help="Number of warmup iterations")
    parser.add_argument("--runs", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' or '0'")
    return parser.parse_args()

def main():
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        # Check yolov5 subfolder fallback
        alt_path = Path("yolov5") / args.weights
        if alt_path.exists():
            weights_path = alt_path
        else:
            raise FileNotFoundError(f"Weights not found at {args.weights}")

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print("=" * 65)
    print("YOLO MODEL PERFORMANCE BENCHMARK")
    print("=" * 65)
    print(f"Weights   : {weights_path}")
    print(f"Device    : {device}")
    print(f"Image Size: {args.img}x{args.img}")
    print(f"Batch Size: {args.batch_size}")
    print("-" * 65)

    # Load model
    model = torch.hub.load("yolov5", "custom", path=str(weights_path), source="local")
    model.to(device)
    model.eval()

    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Dummy tensor
    dummy_input = torch.randn(args.batch_size, 3, args.img, args.img).to(device)

    # Warmup
    print(f"Warming up ({args.warmup} iterations)...")
    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(dummy_input)

    # Timing runs
    print(f"Benchmarking ({args.runs} iterations)...")
    timings = []
    with torch.inference_mode():
        for _ in range(args.runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000)  # ms

    timings = np.array(timings)
    mean_ms = np.mean(timings)
    std_ms = np.std(timings)
    p50_ms = np.percentile(timings, 50)
    p95_ms = np.percentile(timings, 95)
    fps = (1000.0 / mean_ms) * args.batch_size

    print("=" * 65)
    print("BENCHMARK RESULTS")
    print("=" * 65)
    print(f"Total Parameters     : {total_params:,} ({total_params / 1e6:.2f} M)")
    print(f"Trainable Parameters : {trainable_params:,}")
    print(f"Average Latency      : {mean_ms:.2f} ms +/- {std_ms:.2f} ms")
    print(f"Median (P50) Latency : {p50_ms:.2f} ms")
    print(f"95th Percentile (P95): {p95_ms:.2f} ms")
    print(f"Throughput           : {fps:.2f} FPS")
    print("=" * 65)

if __name__ == "__main__":
    main()
