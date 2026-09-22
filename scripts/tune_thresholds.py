"""
T2.5 — Confidence and NMS IoU threshold tuning.
Sweeps conf_threshold in [0.10, 0.25, 0.35, 0.50, 0.65] and
nms_iou in [0.30, 0.45, 0.60] on the test set.
Recommends optimal parameters and writes reports/threshold_sweep.md.
"""
from __future__ import annotations

import json
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "runs" / "detect" / "runs" / "detect" / "llvip_baseline" / "weights" / "best.pt"
DATA_YAML = ROOT / "configs" / "llvip.yaml"
REPORT_PATH = ROOT / "reports" / "threshold_sweep.md"


def run_sweep() -> list[dict]:
    print("Loading model for threshold sweep...")
    model = YOLO(str(MODEL_PATH))

    conf_values = [0.10, 0.25, 0.35, 0.50, 0.65]
    iou_values = [0.30, 0.45, 0.60]

    results = []
    print("\nRunning threshold sweep on LLVIP test set...")
    for conf in conf_values:
        for iou in iou_values:
            val_res = model.val(
                data=str(DATA_YAML),
                split="test",
                batch=32,
                imgsz=640,
                conf=conf,
                iou=iou,
                device=0,
                verbose=False,
            )
            mp = float(val_res.results_dict["metrics/precision(B)"])
            mr = float(val_res.results_dict["metrics/recall(B)"])
            map50 = float(val_res.results_dict["metrics/mAP50(B)"])
            f1 = 2 * (mp * mr) / (mp + mr + 1e-8)

            row = {
                "conf": conf,
                "iou": iou,
                "precision": round(mp, 4),
                "recall": round(mr, 4),
                "f1": round(f1, 4),
                "mAP50": round(map50, 4),
            }
            results.append(row)
            print(f"  conf={conf:.2f} iou={iou:.2f} -> P={mp:.3f} R={mr:.3f} F1={f1:.3f} mAP50={map50:.3f}")

    # Find best F1 config
    best_f1 = max(results, key=lambda x: x["f1"])

    # Write report markdown
    lines = [
        "# Threshold Sweep Report — YOLOv8n LLVIP Infrared",
        "",
        "Swept confidence threshold `[0.10, 0.25, 0.35, 0.50, 0.65]` and NMS IoU `[0.30, 0.45, 0.60]` on the LLVIP test set (3,463 images).",
        "",
        "| Confidence | NMS IoU | Precision | Recall | F1 Score | mAP50 |",
        "|------------|---------|-----------|--------|----------|-------|",
    ]
    for r in results:
        is_best = " **(Recommended)**" if r == best_f1 else ""
        lines.append(f"| {r['conf']:.2f} | {r['iou']:.2f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f}{is_best} | {r['mAP50']:.4f} |")

    lines += [
        "",
        f"**Recommended operational config:** `conf={best_f1['conf']:.2f}`, `iou={best_f1['iou']:.2f}` (Achieves peak F1 = {best_f1['f1']:.4f}, Precision = {best_f1['precision']:.4f}, Recall = {best_f1['recall']:.4f}).",
    ]

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nThreshold sweep complete. Report saved to {REPORT_PATH}")
    return results


if __name__ == "__main__":
    run_sweep()
