"""
T2.2 (revised) — Training YOLOv8n at sensor resolution (160x160).
Uses sensor-emulated LLVIP dataset (160x120 images from prepare_dataset.py).
Trains from the existing 640px checkpoint so fine-tuning converges faster.

Outputs: runs/detect/llvip_sensor_res/weights/best.pt
"""
from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO

ROOT       = Path(__file__).parent.parent
# Fine-tune from existing 640px best.pt (transfer learning at sensor res)
BASE_PT    = ROOT / "runs" / "detect" / "llvip_baseline" / "weights" / "best.pt"
DATA_YAML  = ROOT / "configs" / "llvip.yaml"
RUNS_DIR   = ROOT / "runs" / "detect"


def train() -> None:
    if not BASE_PT.exists():
        print(f"[WARNING] Base checkpoint not found at {BASE_PT}. Training from pretrained yolov8n.pt instead.")
        model = YOLO("yolov8n.pt")
    else:
        print(f"Fine-tuning from {BASE_PT}...")
        model = YOLO(str(BASE_PT))

    print("Training at 160px (sensor resolution)...")
    results = model.train(
        data=str(DATA_YAML),
        epochs=30,
        imgsz=160,          # ← sensor resolution
        batch=64,
        workers=0,
        device=0,
        project=str(RUNS_DIR),
        name="llvip_sensor_res",
        exist_ok=True,
        patience=10,
        optimizer="AdamW",
        lr0=1e-4,           # lower LR for fine-tuning
        lrf=0.01,
        augment=True,
        mosaic=0.5,
        mixup=0.0,
        close_mosaic=5,
        verbose=True,
    )
    print(f"Training complete. Best model: {RUNS_DIR}/llvip_sensor_res/weights/best.pt")
    print(f"Best val mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")


if __name__ == "__main__":
    train()
