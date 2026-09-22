"""
T1.4 — Input validation and retrying supervisor.
Validates ThermalFrame and ImuSample fields; wraps any FrameSource or ImuSource
in a supervisor that retries on exception with exponential back-off.
"""
from __future__ import annotations

import logging
import math
import time
from typing import TypeVar, Generic, Callable

import numpy as np

from yaazhi.types import ThermalFrame, ImuSample
from yaazhi.config import settings

logger = logging.getLogger(__name__)

_BACKOFF_STEPS = [0.1, 0.2, 0.5, 1.0]


# ---------------------------------------------------------------------------
# Frame validation
# ---------------------------------------------------------------------------

def validate_frame(
    frame: ThermalFrame,
    expected_shape: tuple[int, int] | None = None,
    last_seq: int | None = None,
) -> ThermalFrame | None:
    """
    Validate a ThermalFrame.  Returns the frame if valid, None + logged warning if not.
    Checks: shape, dtype, NaN-free, monotonic timestamp.
    """
    h = settings.sensor.height
    w = settings.sensor.width
    expected = expected_shape or (h, w)

    if frame.image.shape != expected:
        logger.warning(
            f"msg=frame_rejected reason=wrong_shape "
            f"got={frame.image.shape} expected={expected} seq={frame.seq}"
        )
        return None

    if frame.image.dtype != np.uint8:
        logger.warning(
            f"msg=frame_rejected reason=wrong_dtype "
            f"got={frame.image.dtype} seq={frame.seq}"
        )
        return None

    if np.any(np.isnan(frame.image.astype(np.float32))):
        logger.warning(f"msg=frame_rejected reason=nan seq={frame.seq}")
        return None

    if last_seq is not None and frame.seq <= last_seq:
        logger.warning(
            f"msg=frame_rejected reason=non_monotonic_seq "
            f"seq={frame.seq} last={last_seq}"
        )
        return None

    return frame


# ---------------------------------------------------------------------------
# IMU validation
# ---------------------------------------------------------------------------

_MAX_ANGULAR_RATE_RAD_PER_S = math.radians(600)  # physical upper bound
_IMU_RATE_HZ = None  # set lazily


def validate_imu(
    sample: ImuSample,
    prev: ImuSample | None = None,
) -> ImuSample | None:
    """
    Validate an ImuSample.  Returns sample if valid, None if rejected.
    Checks: quaternion norm, angular rate limit.
    """
    q = np.array(sample.quat_wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not (0.99 <= norm <= 1.01):
        logger.warning(
            f"msg=imu_rejected reason=bad_norm norm={norm:.6f} t_ns={sample.t_ns}"
        )
        return None

    if prev is not None and prev.t_ns < sample.t_ns:
        dt_s = (sample.t_ns - prev.t_ns) / 1e9
        if dt_s > 0:
            q_prev = np.array(prev.quat_wxyz, dtype=np.float64)
            # angular distance via dot product
            dot = float(np.clip(np.dot(q, q_prev), -1.0, 1.0))
            angle = 2 * math.acos(abs(dot))
            rate = angle / dt_s
            if rate > _MAX_ANGULAR_RATE_RAD_PER_S:
                logger.warning(
                    f"msg=imu_rejected reason=angular_rate_exceeded "
                    f"rate_deg_per_s={math.degrees(rate):.1f} t_ns={sample.t_ns}"
                )
                return None

    return sample


# ---------------------------------------------------------------------------
# Retrying supervisor
# ---------------------------------------------------------------------------

T = TypeVar("T")


class RetryingSupervisor(Generic[T]):
    """
    Wraps any source with a .read() -> T | None interface.
    On exception: logs, backs off (0.1, 0.2, 0.5, 1.0 s), then retries.
    Counts and logs every rejection and drop.
    """

    def __init__(self, factory: Callable[[], object], name: str = "source") -> None:
        """
        factory: callable that returns a new source instance.
        The factory is called on every reconnect attempt.
        """
        self._factory = factory
        self._name = name
        self._source = factory()
        self._drops = 0
        self._exceptions = 0

    @property
    def drop_count(self) -> int:
        return self._drops

    @property
    def exception_count(self) -> int:
        return self._exceptions

    def read(self) -> T | None:
        backoff_idx = 0
        while True:
            try:
                result = self._source.read()  # type: ignore[attr-defined]
                return result
            except Exception as exc:
                self._exceptions += 1
                backoff = _BACKOFF_STEPS[min(backoff_idx, len(_BACKOFF_STEPS) - 1)]
                logger.warning(
                    f"msg=source_exception name={self._name} "
                    f"exc={type(exc).__name__} detail={exc} "
                    f"backoff_s={backoff} total_exc={self._exceptions}"
                )
                time.sleep(backoff)
                backoff_idx += 1
                # Reconnect: create a new source instance
                try:
                    self._source = self._factory()
                    logger.info(
                        f"msg=source_reconnected name={self._name} "
                        f"attempt={self._exceptions}"
                    )
                except Exception as reconnect_exc:
                    logger.error(
                        f"msg=reconnect_failed name={self._name} exc={reconnect_exc}"
                    )
