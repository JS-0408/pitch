"""
T3.1 — Pinhole camera model for the Lepton-class thermal sensor.
Computes intrinsic matrix K from config (width, height, hfov_deg).
Provides pixel-to-ray and ray-to-pixel helpers.
"""
from __future__ import annotations

import math

import numpy as np

from yaazhi.config import settings


def build_K(
    width: int | None = None,
    height: int | None = None,
    hfov_deg: float | None = None,
) -> np.ndarray:
    """
    Build 3x3 pinhole intrinsics matrix K.
    Defaults to values from config if not supplied.
    """
    w = width if width is not None else settings.sensor.width
    h = height if height is not None else settings.sensor.height
    hfov = hfov_deg if hfov_deg is not None else settings.sensor.hfov_deg

    fx = (w / 2.0) / math.tan(math.radians(hfov / 2.0))
    fy = fx  # square pixels assumed for thermal sensor
    cx = w / 2.0
    cy = h / 2.0

    K = np.array([
        [fx,  0., cx],
        [0.,  fy, cy],
        [0.,  0.,  1.],
    ], dtype=np.float64)
    return K


def pixel_to_ray(u: float, v: float, K: np.ndarray) -> np.ndarray:
    """
    Convert pixel (u, v) to unit ray direction in camera coordinates.
    Returns shape (3,).
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
    return ray / np.linalg.norm(ray)


def ray_to_pixel(ray: np.ndarray, K: np.ndarray) -> tuple[float, float]:
    """
    Project a ray (camera coordinates) to pixel (u, v).
    Performs perspective division.
    """
    if ray[2] <= 0:
        raise ValueError("Ray is behind camera (z <= 0)")
    p = K @ ray
    return (p[0] / p[2], p[1] / p[2])
