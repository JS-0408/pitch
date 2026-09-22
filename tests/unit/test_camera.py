# T3.1 — Camera model tests
import math
import numpy as np
from yaazhi.reprojection.camera import build_K, pixel_to_ray, ray_to_pixel


def test_center_pixel_to_optical_axis():
    K = build_K(width=160, height=120, hfov_deg=57.0)
    cx, cy = K[0, 2], K[1, 2]
    ray = pixel_to_ray(cx, cy, K)
    # Should point along z-axis
    assert abs(ray[0]) < 1e-6
    assert abs(ray[1]) < 1e-6
    assert abs(ray[2] - 1.0) < 1e-6


def test_round_trip_pixel_ray():
    K = build_K(width=160, height=120, hfov_deg=57.0)
    for u, v in [(0, 0), (80, 60), (159, 119), (40, 30)]:
        ray = pixel_to_ray(float(u), float(v), K)
        u2, v2 = ray_to_pixel(ray * 5.0, K)  # scale doesn't matter
        assert abs(u2 - u) < 1e-6, f"u round-trip error at ({u},{v}): {abs(u2-u)}"
        assert abs(v2 - v) < 1e-6, f"v round-trip error at ({u},{v}): {abs(v2-v)}"


def test_focal_length_from_fov():
    w, h, hfov = 160, 120, 57.0
    K = build_K(width=w, height=h, hfov_deg=hfov)
    expected_fx = (w / 2.0) / math.tan(math.radians(hfov / 2.0))
    assert abs(K[0, 0] - expected_fx) < 1e-9
