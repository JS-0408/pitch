# T3.2 — Warp tests
import math
import numpy as np
from yaazhi.reprojection.camera import build_K
from yaazhi.reprojection.warp import reproject, rotation_delta


def _identity_quat():
    return (1.0, 0.0, 0.0, 0.0)


def _yaw_quat(deg: float):
    """Pure yaw rotation quaternion."""
    rad = math.radians(deg)
    return (math.cos(rad/2), 0.0, math.sin(rad/2), 0.0)


def test_zero_delta_is_identity():
    K = build_K(width=160, height=120, hfov_deg=57.0)
    img = np.random.randint(0, 255, (120, 160), dtype=np.uint8)
    q = _identity_quat()
    R = rotation_delta(q, q)
    result = reproject(img, K, R, max_delta_deg=60)
    assert not result.clamped
    diff = img.astype(np.int16) - result.warped.astype(np.int16)
    assert np.mean(np.abs(diff)) < 2.0, "Zero delta should be near-identity"


def test_yaw_shifts_center():
    """A 10 deg yaw should shift the image center by roughly f_px * tan(10 deg) pixels."""
    w, h, hfov = 160, 120, 57.0
    K = build_K(width=w, height=h, hfov_deg=hfov)
    f_px = K[0, 0]
    expected_shift = f_px * math.tan(math.radians(10))  # pixels

    q_from = _identity_quat()
    q_to = _yaw_quat(10)
    R = rotation_delta(q_from, q_to)

    # Warp an image with a bright dot in the center
    img = np.zeros((h, w), dtype=np.uint8)
    img[h//2, w//2] = 255

    result = reproject(img, K, R, max_delta_deg=60)
    assert not result.clamped

    # Find peak in warped image
    peak = np.unravel_index(np.argmax(result.warped), result.warped.shape)
    actual_shift = abs(peak[1] - w // 2)
    assert abs(actual_shift - expected_shift) <= 2.0, (
        f"Expected shift ~{expected_shift:.1f}px, got {actual_shift}px"
    )


def test_clamped_large_rotation():
    K = build_K(width=160, height=120, hfov_deg=57.0)
    img = np.zeros((120, 160), dtype=np.uint8)
    q_from = _identity_quat()
    q_to = _yaw_quat(90)
    R = rotation_delta(q_from, q_to)
    result = reproject(img, K, R, max_delta_deg=60)
    assert result.clamped
