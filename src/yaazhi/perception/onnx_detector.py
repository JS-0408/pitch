"""
T2.4 — ONNX Detector Wrapper.
Inference wrapper around ONNX Runtime for YOLOv8n.
Accepts uint8 thermal frame (HxW or HxW3), applies NMS,
returns list[Detection] dataclass objects.
Supports CPU and CUDA execution providers seamlessly.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from yaazhi.config import settings
from yaazhi.types import Detection

logger = logging.getLogger(__name__)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Pure numpy/OpenCV non-maximum suppression."""
    if len(boxes) == 0:
        return []
    # boxes format: (x1, y1, x2, y2)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


class OnnxDetector:
    """
    ONNX Runtime detector for Yaazhi thermal person detection.
    Preprocesses frame → 640x640 float32 → ONNX inference → NMS → list[Detection].
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf_thresh: float | None = None,
        iou_thresh: float | None = None,
    ) -> None:
        if model_path is None:
            # Fallback check
            p1 = Path("models/yolov8n_llvip.onnx")
            p2 = Path("runs/detect/runs/detect/llvip_baseline/weights/best.onnx")
            model_path = p1 if p1.exists() else p2

        self._model_path = Path(model_path)
        self._conf_thresh = conf_thresh if conf_thresh is not None else float(getattr(settings.detector, "conf", 0.25))
        self._iou_thresh = iou_thresh if iou_thresh is not None else float(getattr(settings.detector, "nms_iou", 0.5))

        # Initialize ONNX Runtime session
        avail_providers = ort.get_available_providers()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in avail_providers else ["CPUExecutionProvider"]
        
        try:
            self._session = ort.InferenceSession(str(self._model_path), providers=providers)
        except Exception:
            # Fallback to pure CPU if CUDA provider DLL fails
            self._session = ort.InferenceSession(str(self._model_path), providers=["CPUExecutionProvider"])

        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        input_shape = self._session.get_inputs()[0].shape
        self._img_h = input_shape[2] if isinstance(input_shape[2], int) else 160
        self._img_w = input_shape[3] if isinstance(input_shape[3], int) else 160

        logger.info(
            f"msg=onnx_detector_initialized model={self._model_path.name} "
            f"input_shape=({self._img_h},{self._img_w}) provider={self._session.get_providers()[0]} conf={self._conf_thresh} iou={self._iou_thresh}"
        )

    def detect(self, image: np.ndarray) -> list[Detection]:
        """
        Run detection on image (uint8 grayscale HxW or BGR HxWx3).
        Returns list of Detection(xyxy, conf, cls=0).
        """
        h_orig, w_orig = image.shape[:2]

        # Convert grayscale to 3-channel if needed
        if image.ndim == 2:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = image

        # Resize to model input shape (self._img_w, self._img_h)
        img_resized = cv2.resize(img_bgr, (self._img_w, self._img_h))

        # Preprocess: HWC BGR → NCHW RGB float32 [0, 1]
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        tensor = img_rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, :]  # (1, 3, H, W)

        # Inference
        outputs = self._session.run([self._output_name], {self._input_name: tensor})[0]
        # shape: (1, 5, 8400) -> [cx, cy, w, h, score]

        preds = outputs[0]  # (5, 8400)
        cx, cy, w, h, scores = preds[0], preds[1], preds[2], preds[3], preds[4]

        # Filter by confidence threshold
        mask = scores >= self._conf_thresh
        if not np.any(mask):
            return []

        cx, cy, w, h, scores = cx[mask], cy[mask], w[mask], h[mask], scores[mask]

        # Convert (cx, cy, w, h) → (x1, y1, x2, y2) in 640x640 space
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        boxes_640 = np.column_stack([x1, y1, x2, y2])

        # Apply Non-Maximum Suppression (NMS)
        keep = _nms(boxes_640, scores, self._iou_thresh)

        scale_x = w_orig / float(self._img_w)
        scale_y = h_orig / float(self._img_h)

        detections = []
        for i in keep:
            bx1 = float(boxes_640[i, 0] * scale_x)
            by1 = float(boxes_640[i, 1] * scale_y)
            bx2 = float(boxes_640[i, 2] * scale_x)
            by2 = float(boxes_640[i, 3] * scale_y)

            # Clamp to image bounds
            bx1 = max(0.0, min(float(w_orig), bx1))
            by1 = max(0.0, min(float(h_orig), by1))
            bx2 = max(0.0, min(float(w_orig), bx2))
            by2 = max(0.0, min(float(h_orig), by2))

            detections.append(
                Detection(
                    xyxy=(bx1, by1, bx2, by2),
                    conf=float(scores[i]),
                    cls=0,
                )
            )

        return detections
