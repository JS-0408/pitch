"""
T3.2 — Rotation-delta perspective warp.
H = K @ R_delta @ K^-1; warp via cv2.warpPerspective; return valid-region mask.
Clamps |delta| to max_delta_deg; reports a flag if clamped.
"""
from __future__ import annotations

import logging
import math
from typing import NamedTuple

import cv2
import numpy as np

from yaazhi.config import settings
from yaazhi.reprojection.camera import build_K

logger = logging.getLogger(__name__)

_K: np.ndarray | None = None


def _get_K() -> np.ndarray:
    global _K
    if _K is None:
        _K = build_K()
    return _K


def rotation_delta(q_from: tuple, q_to: tuple) -> np.ndarray:
    """
    Compute 3x3 rotation matrix R_delta = R_to @ R_from^T.
    Inputs are quaternions (w, x, y, z).
    """
    def quat_to_rot(q: tuple) -> np.ndarray:
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ], dtype=np.float64)

    R_from = quat_to_rot(q_from)
    R_to = quat_to_rot(q_to)
    return R_to @ R_from.T


class WarpResult(NamedTuple):
    warped: np.ndarray        # warped image (HxW uint8)
    valid_mask: np.ndarray    # HxW bool; True where warp is valid
    clamped: bool             # True if delta exceeded max_delta_deg


def reproject(
    image: np.ndarray,
    K: np.ndarray,
    R_delta: np.ndarray,
    max_delta_deg: float | None = None,
) -> WarpResult:
    """
    Warp image by rotation delta using the planar homography:
       H = K @ R_delta @ K^-1
    Returns (warped_image, valid_mask, clamped_flag).
    Black borders where invalid (no smearing).
    """
    max_deg = max_delta_deg
    if max_deg is None:
        max_deg = settings.reprojection.max_delta_deg

    # Check rotation angle
    trace = np.clip((np.trace(R_delta) - 1) / 2, -1.0, 1.0)
    angle_deg = math.degrees(math.acos(trace))
    clamped = False

    if angle_deg > max_deg:
        clamped = True
        logger.warning(
            f"msg=reprojection_clamped angle_deg={angle_deg:.1f} max_deg={max_deg}"
        )
        return WarpResult(image.copy(), np.ones(image.shape[:2], dtype=bool), clamped)

    K_inv = np.linalg.inv(K)
    H = K @ R_delta @ K_inv
    H /= H[2, 2]

    h, w = image.shape[:2]
    warped = cv2.warpPerspective(
        image, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Build valid-region mask by warping a white image
    white = np.ones((h, w), dtype=np.uint8) * 255
    mask_raw = cv2.warpPerspective(
        white, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid_mask = mask_raw > 127

    return WarpResult(warped, valid_mask, clamped)
