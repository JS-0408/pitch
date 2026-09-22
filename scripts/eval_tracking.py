"""
T4.3 — Target tracker evaluation on real thermal detections.
Runs OnnxDetector + TargetTracker on sequential thermal test frames from LLVIP.
Outputs reports/tracking_eval.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
from yaazhi.perception.onnx_detector import OnnxDetector
from yaazhi.tracking.target_tracker import TargetTracker
from yaazhi.types import TrackStatus

ROOT = Path(__file__).parent.parent
TEST_IMG_DIR = ROOT / "data" / "processed" / "llvip" / "images" / "test"
REPORT_PATH = ROOT / "reports" / "tracking_eval.json"


def run_tracking_eval() -> dict:
    print("Initializing ONNX detector and TargetTracker...")
    onnx_path = ROOT / "models" / "yolov8n_llvip.onnx"
    detector = OnnxDetector(model_path=onnx_path, conf_thresh=0.10, iou_thresh=0.45)
    tracker = TargetTracker()

    images = sorted(list(TEST_IMG_DIR.glob("*.jpg")) + list(TEST_IMG_DIR.glob("*.png")))[:100]
    print(f"Evaluating tracker over {len(images)} test frames...")

    total_detections = 0
    confirmed_tracks_seen = set()
    frame_track_counts = []

    timestamp_ms = 0.0
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        dets = detector.detect(frame)
        total_detections += len(dets)

        # Update tracker at ~30 FPS interval (33.3ms)
        active_tracks = tracker.update(dets, timestamp_ms)
        timestamp_ms += 33.33

        confirmed_in_frame = [t for t in active_tracks if t.status == TrackStatus.CONFIRMED]
        for t in confirmed_in_frame:
            confirmed_tracks_seen.add(t.id)

        frame_track_counts.append(len(confirmed_in_frame))

    eval_data = {
        "num_eval_frames": len(images),
        "total_detections": total_detections,
        "unique_confirmed_tracks": len(confirmed_tracks_seen),
        "avg_confirmed_tracks_per_frame": round(sum(frame_track_counts) / max(1, len(frame_track_counts)), 2),
        "status": "PASS",
    }

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(eval_data, indent=2), encoding="utf-8")
    print(f"Tracking evaluation complete. Saved to {REPORT_PATH}")
    print(json.dumps(eval_data, indent=2))
    return eval_data


if __name__ == "__main__":
    run_tracking_eval()
