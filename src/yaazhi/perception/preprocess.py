"""
T5.1 — Thermal preprocessing / AGC (Automatic Gain Control).
Three modes: naive, percentile, adaptive (CLAHE with hot-pixel exclusion).
Also provides inject_hot_blob() for testing.
"""
from __future__ import annotations

import logging
from enum import Enum

import cv2
import numpy as np

from yaazhi.config import settings

logger = logging.getLogger(__name__)


class AgcMode(Enum):
    NAIVE = "naive"
    PERCENTILE = "percentile"
    ADAPTIVE = "adaptive"


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Stretch float32 [min,max] to uint8 [0,255]."""
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).clip(0, 255).astype(np.uint8)


def preprocess(image: np.ndarray, mode: AgcMode | None = None) -> np.ndarray:
    """
    Apply AGC to a uint8 thermal image.
    Returns uint8 image ready for display / detector.

    Modes:
      naive       - simple min-max stretch
      percentile  - clip to [1%, 99.5%] then stretch
      adaptive    - percentile + CLAHE with hot-pixel exclusion so a
                    saturated blob does not compress person contrast
    """
    if mode is None:
        raw_mode = settings.preprocess.get("agc", "adaptive")
        mode = AgcMode(raw_mode)

    img = image.astype(np.float32)

    if mode == AgcMode.NAIVE:
        return _normalize_to_uint8(img)

    elif mode == AgcMode.PERCENTILE:
        lo = float(np.percentile(img, 1.0))
        hi = float(np.percentile(img, 99.5))
        clipped = np.clip(img, lo, hi)
        return _normalize_to_uint8(clipped)

    else:  # ADAPTIVE
        # Step 1: identify hot pixels (top 0.5%) and exclude from range calc
        hot_thresh = float(np.percentile(img, 99.5))
        mask_normal = img < hot_thresh

        if mask_normal.sum() > 100:
            lo = float(np.percentile(img[mask_normal], 1.0))
            hi = float(np.percentile(img[mask_normal], 99.5))
        else:
            lo = float(img.min())
            hi = float(img.max())

        clipped = np.clip(img, lo, hi)
        norm = _normalize_to_uint8(clipped)

        # Step 2: CLAHE on the stretch-corrected image
        clip_limit = float(settings.preprocess.get("clahe_clip", 2.0))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(4, 4))
        enhanced = clahe.apply(norm)

        return enhanced


def inject_hot_blob(
    image: np.ndarray,
    size: int = 10,
    temp_value: int = 255,
    cx: int | None = None,
    cy: int | None = None,
) -> np.ndarray:
    """
    Inject a saturated circular blob into image (simulates a hot object).
    Used for testing AGC hot-scene handling.
    """
    out = image.copy()
    h, w = out.shape[:2]
    cx = cx if cx is not None else w // 2
    cy = cy if cy is not None else h // 4
    cv2.circle(out, (cx, cy), size // 2, int(temp_value), -1)
    return out


def local_rms_contrast(image: np.ndarray, boxes: list[tuple]) -> list[float]:
    """
    Compute RMS contrast in each bounding box region.
    boxes: list of (x1, y1, x2, y2) ints.
    Returns list of RMS values.
    """
    results = []
    for (x1, y1, x2, y2) in boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(image.shape[1], int(x2)), min(image.shape[0], int(y2))
        region = image[y1:y2, x1:x2].astype(np.float32)
        if region.size > 0:
            rms = float(np.sqrt(np.mean((region - region.mean()) ** 2)))
        else:
            rms = 0.0
        results.append(rms)
    return results
