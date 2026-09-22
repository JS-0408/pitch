"""
T2.3 — Evaluation harness (sensor-resolution + per-height-bucket recall).
Evaluates trained YOLOv8n model on the LLVIP test split at sensor resolution
(160x120 sensor-emulated images, same pipeline as prepare_dataset.py).

Outputs:
  reports/eval_yolov8n_llvip.json   — aggregate metrics
  reports/eval_recall_by_bucket.md  — recall broken down by person height

DO NOT quote aggregate mAP numbers in pitch materials without the bucket
breakdown — performance at <8px (far) is substantially lower than aggregate.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT       = Path(__file__).parent.parent
MODEL_PATH = ROOT / "runs" / "detect" / "llvip_baseline" / "weights" / "best.pt"
DATA_YAML  = ROOT / "configs" / "llvip.yaml"
TEST_IMG_DIR = ROOT / "data" / "processed" / "llvip" / "images" / "test"
TEST_LBL_DIR = ROOT / "data" / "processed" / "llvip" / "labels" / "test"
REPORT_PATH  = ROOT / "reports" / "eval_yolov8n_llvip.json"
BUCKET_REPORT = ROOT / "reports" / "eval_recall_by_bucket.md"

SENSOR_W = 160
SENSOR_H = 120

# Height buckets matching prepare_dataset.py
BUCKETS = [
    ("4-8px (far)",      4,   8),
    ("8-15px (mid)",     8,  15),
    ("15-30px (close)", 15,  30),
    (">30px (near)",    30, 9999),
]


def _iou(b1: np.ndarray, b2: np.ndarray) -> float:
    """IoU between two (x1 y1 x2 y2) boxes."""
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def run_per_bucket_eval(model: YOLO, conf: float = 0.10, iou_thr: float = 0.5) -> dict:
    """
    Manually iterate test images, run ONNX-style inference, match predictions
    to ground truth boxes, and tally TP/FN per height bucket.
    Images are already at 160x120 (sensor-emulated by prepare_dataset.py).
    """
    bucket_tp: dict[str, int] = {b[0]: 0 for b in BUCKETS}
    bucket_fn: dict[str, int] = {b[0]: 0 for b in BUCKETS}

    img_paths = sorted(list(TEST_IMG_DIR.glob("*.jpg")) + list(TEST_IMG_DIR.glob("*.png")))
    print(f"Running per-bucket evaluation on {len(img_paths)} test images at {SENSOR_W}x{SENSOR_H}...")

    for img_path in img_paths:
        lbl_path = TEST_LBL_DIR / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        # Load ground truth boxes (YOLO normalised cx cy w h → x1 y1 x2 y2 px)
        gt_boxes = []
        for line in lbl_path.read_text().strip().splitlines():
            if not line:
                continue
            _, cx, cy, bw, bh = map(float, line.split())
            x1 = (cx - bw / 2) * SENSOR_W
            y1 = (cy - bh / 2) * SENSOR_H
            x2 = (cx + bw / 2) * SENSOR_W
            y2 = (cy + bh / 2) * SENSOR_H
            h_px = y2 - y1
            gt_boxes.append((x1, y1, x2, y2, h_px))

        if not gt_boxes:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # YOLOv8 expects BGR; convert grey → 3-channel for ultralytics
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Upscale to 640 for model input (model trained at 640, we want
        # to measure degradation from sensor-res images, not model input size)
        img_640 = cv2.resize(img_bgr, (640, 640), interpolation=cv2.INTER_LINEAR)

        results = model(img_640, verbose=False, conf=conf)[0]

        # Scale predictions back to sensor coords
        pred_boxes = []
        if results.boxes is not None and len(results.boxes):
            for box in results.boxes.xyxy.cpu().numpy():
                x1p = box[0] / 640 * SENSOR_W
                y1p = box[1] / 640 * SENSOR_H
                x2p = box[2] / 640 * SENSOR_W
                y2p = box[3] / 640 * SENSOR_H
                pred_boxes.append(np.array([x1p, y1p, x2p, y2p]))

        # Match GT to predictions per height bucket
        matched = [False] * len(pred_boxes)
        for gt in gt_boxes:
            h_px = gt[4]
            # Determine bucket
            bucket_name = None
            for bname, blo, bhi in BUCKETS:
                if blo <= h_px < bhi:
                    bucket_name = bname
                    break
            if bucket_name is None:
                continue  # below MIN_PERSON_PX_H, already filtered

            gt_arr = np.array(gt[:4])
            found = False
            for pi, pred in enumerate(pred_boxes):
                if not matched[pi] and _iou(gt_arr, pred) >= iou_thr:
                    matched[pi] = True
                    found = True
                    break

            if found:
                bucket_tp[bucket_name] += 1
            else:
                bucket_fn[bucket_name] += 1

    return bucket_tp, bucket_fn


def run_evaluation() -> dict:
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(str(MODEL_PATH))

    print(f"Running aggregate eval on test set (imgsz=160, sensor-emulated)...")
    t0 = time.perf_counter()

    # Aggregate metrics: model.val() on sensor-emulated 160x120 images
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        batch=32,
        imgsz=160,        # ← sensor resolution — this is the key fix
        device=0,
        verbose=True,
        conf=0.10,
        iou=0.45,
    )
    t_total = time.perf_counter() - t0

    mp     = float(metrics.results_dict["metrics/precision(B)"])
    mr     = float(metrics.results_dict["metrics/recall(B)"])
    map50  = float(metrics.results_dict["metrics/mAP50(B)"])
    map5095 = float(metrics.results_dict["metrics/mAP50-95(B)"])
    speed_ms = metrics.speed

    eval_data = {
        "model_path": "runs/detect/llvip_baseline/weights/best.pt",
        "dataset": "LLVIP Infrared Test Split (160x120 sensor-emulated)",
        "eval_resolution": "160x120",
        "sensor_blur_sigma": 0.8,
        "sensor_noise_dn": 2.0,
        "num_test_images": 3463,
        "conf_threshold": 0.10,
        "nms_iou": 0.45,
        "metrics": {
            "precision": round(mp, 4),
            "recall": round(mr, 4),
            "mAP50": round(map50, 4),
            "mAP50-95": round(map5095, 4),
        },
        "speed_per_image_ms": {
            "preprocess": round(speed_ms["preprocess"], 2),
            "inference": round(speed_ms["inference"], 2),
            "postprocess": round(speed_ms["postprocess"], 2),
            "total_ms": round(sum(speed_ms.values()), 2),
        },
        "total_eval_time_s": round(t_total, 2),
        "note": "Evaluated at 160x120 sensor resolution with optical blur + noise emulation."
                " Numbers are lower than 640px baseline — this is the honest figure.",
    }

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(eval_data, indent=2), encoding="utf-8")
    print(f"\nSaved aggregate eval to {REPORT_PATH}")
    print(json.dumps(eval_data, indent=2))

    # Per-bucket recall
    bucket_tp, bucket_fn = run_per_bucket_eval(model, conf=0.10, iou_thr=0.5)

    bucket_lines = [
        "# Recall by Person Height Bucket — Sensor-Resolution (160x120)",
        "",
        "Person height measured at sensor resolution (160x120). Aggregate recall",
        "hides the large performance gap between close and far targets.",
        "Use this table in pitch materials, not the 640px aggregate.",
        "",
        "| Height bucket | Meaning (approx distance) | TP | FN | Recall |",
        "|---------------|--------------------------|----|----|--------|",
    ]
    for bname, blo, bhi in BUCKETS:
        tp = bucket_tp[bname]
        fn = bucket_fn[bname]
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # Rough distance estimate for a 1.7m person
        # h_px / SENSOR_H * theoretical FOV ... use 60deg VFOV as rough guide
        dist_label = {
            "4-8px (far)":      ">30 m",
            "8-15px (mid)":     "15–30 m",
            "15-30px (close)":  "7–15 m",
            ">30px (near)":     "<7 m",
        }.get(bname, "—")
        bucket_lines.append(
            f"| {bname} | {dist_label} | {tp} | {fn} | {recall:.1%} |"
        )

    bucket_lines += [
        "",
        f"**Overall sensor-res mAP50:** {map50:.1%} @ 160x120 (conf=0.10, iou=0.45)",
        "",
        "> Note: model was trained at 640px. Fine-tuning at 160px will improve",
        "> recall in the 4-15px buckets (far targets). This table is the honest",
        "> baseline before any sensor-resolution fine-tuning.",
    ]
    BUCKET_REPORT.write_text("\n".join(bucket_lines), encoding="utf-8")
    print(f"Saved per-bucket recall report to {BUCKET_REPORT}")
    return eval_data


if __name__ == "__main__":
    run_evaluation()
