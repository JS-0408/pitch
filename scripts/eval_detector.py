"""
T2.3 — Evaluation harness.
Evaluates trained YOLOv8n model on the LLVIP test split (3,463 images).
Calculates precision, recall, mAP50, mAP50-95, and per-image inference speed.
Outputs reports/eval_yolov8n_llvip.json and updates tracker.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "runs" / "detect" / "runs" / "detect" / "llvip_baseline" / "weights" / "best.pt"
DATA_YAML = ROOT / "configs" / "llvip.yaml"
REPORT_PATH = ROOT / "reports" / "eval_yolov8n_llvip.json"


def run_evaluation() -> dict:
    print(f"Loading best model from {MODEL_PATH}...")
    model = YOLO(str(MODEL_PATH))

    print(f"Evaluating on test set using {DATA_YAML}...")
    t0 = time.perf_counter()
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        batch=16,
        imgsz=640,
        device=0,
        verbose=True,
    )
    t_total = time.perf_counter() - t0

    # Extract metrics dict
    mp = float(metrics.results_dict["metrics/precision(B)"])
    mr = float(metrics.results_dict["metrics/recall(B)"])
    map50 = float(metrics.results_dict["metrics/mAP50(B)"])
    map50_95 = float(metrics.results_dict["metrics/mAP50-95(B)"])
    speed_ms = metrics.speed  # dict with preprocess, inference, loss, postprocess ms

    eval_data = {
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "dataset": "LLVIP Infrared Test Split",
        "num_test_images": 3463,
        "metrics": {
            "precision": round(mp, 4),
            "recall": round(mr, 4),
            "mAP50": round(map50, 4),
            "mAP50-95": round(map50_95, 4),
        },
        "speed_per_image_ms": {
            "preprocess": round(speed_ms["preprocess"], 2),
            "inference": round(speed_ms["inference"], 2),
            "postprocess": round(speed_ms["postprocess"], 2),
            "total_ms": round(sum(speed_ms.values()), 2),
        },
        "total_eval_time_s": round(t_total, 2),
    }

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(eval_data, indent=2), encoding="utf-8")
    print(f"\nSaved evaluation results to {REPORT_PATH}")
    print(json.dumps(eval_data, indent=2))
    return eval_data


if __name__ == "__main__":
    run_evaluation()
