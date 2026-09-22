"""
T6.1 — Ironbow palette lookup table.
Maps grayscale [0, 255] → BGR (ironbow thermal colormap).
"""
from __future__ import annotations

import numpy as np


def _build_ironbow_lut() -> np.ndarray:
    """
    Hand-tuned ironbow palette: black → blue → red → orange → yellow → white.
    Returns shape (256, 3) uint8 in BGR order for OpenCV.
    """
    stops = [
        (0,   (0,   0,   0)),    # black
        (64,  (128, 0,   64)),   # dark purple
        (96,  (255, 0,   0)),    # blue
        (128, (255, 64,  0)),    # cyan-blue
        (160, (0,   128, 200)),  # green-teal
        (192, (0,   200, 255)),  # yellow-green
        (224, (0,   255, 255)),  # yellow
        (255, (255, 255, 255)),  # white
    ]
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(len(stops) - 1):
        v0, c0 = stops[i]
        v1, c1 = stops[i + 1]
        for v in range(v0, v1 + 1):
            t = (v - v0) / (v1 - v0) if v1 != v0 else 0
            for ch in range(3):
                lut[v, ch] = int(c0[ch] + t * (c1[ch] - c0[ch]))
    return lut


IRONBOW_LUT: np.ndarray = _build_ironbow_lut()  # shape (256,3) uint8 BGR


def apply_ironbow(gray: np.ndarray) -> np.ndarray:
    """
    Apply ironbow colormap to grayscale image.
    gray: HxW uint8.
    Returns HxW×3 uint8 BGR.
    """
    return IRONBOW_LUT[gray]
