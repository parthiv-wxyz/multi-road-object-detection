"""
run_all_tests.py

Unified test suite runner for Enhanced YOLOv5 Multi-Road Object Detection.
Runs all component and architecture tests, reporting execution time and status.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TESTS = [
    "test_eca.py",
    "test_afp.py",
    "test_lwconv.py",
    "test_lwc3.py",
    "test_wise_iou.py",
    "test_cutmix.py",
    "test_refined_anchor.py",
    "test_pipeline_modules.py",
    "test_backend_api.py",
    "test_p2_model.py",
    "test_eca_model.py",
    "test_afp_model.py",
    "test_lwc3_model.py",
]

def main():
    print("=" * 70)
    print("RUNNING COMPLETE ENHANCED YOLOV5 TEST SUITE")
    print("=" * 70)

    results = []
    overall_start = time.perf_counter()

    for test_file in TESTS:
        test_path = ROOT / test_file
        if not test_path.exists():
            print(f"[MISSING] {test_file}")
            results.append((test_file, "MISSING", 0.0, "File not found"))
            continue

        start = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(test_path)],
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - start

        if proc.returncode == 0:
            print(f"[PASS] {test_file:<24} ({elapsed:.2f}s)")
            results.append((test_file, "PASS", elapsed, ""))
        else:
            print(f"[FAIL] {test_file:<24} ({elapsed:.2f}s)")
            results.append((test_file, "FAIL", elapsed, proc.stderr.strip() or proc.stdout.strip()))

    total_time = time.perf_counter() - overall_start
    print("=" * 70)
    passed = sum(1 for r in results if r[1] == "PASS")
    total = len(results)
    print(f"Summary: {passed}/{total} tests passed in {total_time:.2f}s")
    print("=" * 70)

    if passed < total:
        for name, status, _, err in results:
            if status != "PASS":
                print(f"\n--- Failure Details: {name} ---")
                print(err[:500])
        sys.exit(1)

if __name__ == "__main__":
    main()
