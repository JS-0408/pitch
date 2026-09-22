# T2.4 — Unit tests for OnnxDetector wrapper
import numpy as np
import pytest
from pathlib import Path
from yaazhi.perception.onnx_detector import OnnxDetector, _nms
from yaazhi.types import Detection


def test_nms_basic():
    boxes = np.array([
        [10, 10, 50, 50],
        [12, 12, 52, 52],  # high IoU with box 0
        [100, 100, 150, 150],  # no overlap
    ], dtype=np.float32)
    scores = np.array([0.9, 0.85, 0.95], dtype=np.float32)
    keep = _nms(boxes, scores, iou_threshold=0.5)
    assert len(keep) == 2
    assert 2 in keep  # highest score
    assert 0 in keep  # second highest score, box 1 suppressed


def test_onnx_detector_initialization():
    onnx_path = Path("models/yolov8n_llvip.onnx")
    if not onnx_path.exists():
        pytest.skip("ONNX model file not found")

    detector = OnnxDetector(model_path=onnx_path, conf_thresh=0.25, iou_thresh=0.45)
    assert detector is not None


def test_onnx_detector_inference_blank_image():
    onnx_path = Path("models/yolov8n_llvip.onnx")
    if not onnx_path.exists():
        pytest.skip("ONNX model file not found")

    detector = OnnxDetector(model_path=onnx_path, conf_thresh=0.5)
    blank = np.zeros((480, 640), dtype=np.uint8)
    dets = detector.detect(blank)
    assert isinstance(dets, list)
    # Blank black image should return 0 or very few person detections
    assert len(dets) <= 2
