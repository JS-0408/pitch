"""
T5.2 — Smoke simulator.
Applies density-controlled haze to a visible channel (strong degradation)
while giving the thermal channel only mild attenuation + noise.
Labels output clearly in the UI: SIMULATED SMOKE.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from yaazhi.config import settings

logger = logging.getLogger(__name__)


def apply_smoke_visible(
    image: np.ndarray,
    density: float = 0.6,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate smoke on a visible-spectrum image.
    density 0 = no effect, 1 = fully obscured.

    Effects:
     - Low-frequency noise (volumetric haze texture)
     - Global contrast reduction
     - Gaussian blur
     Returns uint8 image.
    """
    if density == 0.0:
        return image.copy()

    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    img = image.astype(np.float32)

    # Low-frequency noise: generate small, then upscale
    small_h, small_w = max(4, h // 8), max(4, w // 8)
    noise_small = rng.uniform(0, 255 * density, (small_h, small_w)).astype(np.float32)
    noise = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_CUBIC)

    # Blend image with haze
    haze_strength = density * 0.7
    img = img * (1 - haze_strength) + noise * haze_strength

    # Contrast reduction
    contrast_factor = 1.0 - density * 0.5
    mean = img.mean()
    img = (img - mean) * contrast_factor + mean

    # Blur
    ksize = max(3, int(density * 10) | 1)  # odd
    img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    return img.clip(0, 255).astype(np.uint8)


def apply_smoke_thermal(
    image: np.ndarray,
    density: float = 0.6,
    seed: int | None = None,
) -> np.ndarray:
    """
    Thermal sees through smoke: only mild noise + slight attenuation.
    Thermal smoke effect is intentionally minor.
    """
    if density == 0.0:
        return image.copy()

    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    img = image.astype(np.float32)

    # Very light Gaussian noise
    noise_sigma = density * 8.0
    noise = rng.normal(0, noise_sigma, (h, w)).astype(np.float32)
    img = img + noise

    # Minimal attenuation
    attenuation = 1.0 - density * 0.05
    img = img * attenuation

    return img.clip(0, 255).astype(np.uint8)


class SmokeSim:
    """
    Stateful smoke simulator — reads config, applies per-frame.
    Apply .thermal(frame) and optionally .visible(frame) each tick.
    """

    def __init__(self) -> None:
        self._enabled = bool(settings.smoke_sim.get("enabled", False))
        self._density = float(settings.smoke_sim.get("density", 0.6))
        self._frame = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def toggle(self) -> bool:
        self._enabled = not self._enabled
        logger.info(f"msg=smoke_sim_toggled enabled={self._enabled}")
        return self._enabled

    def thermal(self, image: np.ndarray) -> np.ndarray:
        if not self._enabled:
            return image
        out = apply_smoke_thermal(image, density=self._density, seed=self._frame)
        self._frame += 1
        return out

    def visible(self, image: np.ndarray) -> np.ndarray:
        if not self._enabled:
            return image
        return apply_smoke_visible(image, density=self._density, seed=self._frame)
