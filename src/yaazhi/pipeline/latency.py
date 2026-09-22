"""
T3.4 — Latency accounting.
Tags each displayed frame with timing metadata.
Computes motion-to-display, frame-age, and rolling stats.
Software-only: excludes display hardware latency.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

_WINDOW = 300  # rolling window (frames)


@dataclass
class FrameTimings:
    """Timing tags attached to every rendered frame."""
    t_frame_capture_ns: int   # when thermal frame was captured by sensor
    t_imu_used_ns: int        # timestamp of the IMU sample used for warp
    t_display_ns: int         # when this frame was handed to OpenCV imshow


@dataclass
class LatencyStats:
    motion_to_display_median_ms: float   # t_display - t_imu_used
    motion_to_display_p95_ms: float
    frame_age_median_ms: float           # t_display - t_frame_capture
    frame_age_p95_ms: float
    n_samples: int


class LatencyTracker:
    """
    Receives FrameTimings on every render tick.
    Reports rolling median / p95 over the last _WINDOW frames.

    Note: motion-to-display is software-only.
    Display hardware latency (panel, HDMI round-trip) is excluded
    because it cannot be measured in software.
    """

    def __init__(self, window: int = _WINDOW) -> None:
        self._m2d: deque[float] = deque(maxlen=window)   # motion-to-display ms
        self._age: deque[float] = deque(maxlen=window)   # frame age ms

    def record(self, timings: FrameTimings) -> None:
        m2d = (timings.t_display_ns - timings.t_imu_used_ns) / 1e6
        age = (timings.t_display_ns - timings.t_frame_capture_ns) / 1e6
        self._m2d.append(m2d)
        self._age.append(age)

    def stats(self) -> LatencyStats | None:
        if not self._m2d:
            return None
        m2d = np.array(self._m2d)
        age = np.array(self._age)
        return LatencyStats(
            motion_to_display_median_ms=float(np.median(m2d)),
            motion_to_display_p95_ms=float(np.percentile(m2d, 95)),
            frame_age_median_ms=float(np.median(age)),
            frame_age_p95_ms=float(np.percentile(age, 95)),
            n_samples=len(m2d),
        )

    def log_stats(self) -> None:
        s = self.stats()
        if s is None:
            return
        logger.info(
            f"msg=latency_stats "
            f"m2d_median_ms={s.motion_to_display_median_ms:.1f} "
            f"m2d_p95_ms={s.motion_to_display_p95_ms:.1f} "
            f"frame_age_median_ms={s.frame_age_median_ms:.1f} "
            f"frame_age_p95_ms={s.frame_age_p95_ms:.1f} "
            f"n={s.n_samples} "
            f"note=software_only_excludes_display_hardware"
        )
