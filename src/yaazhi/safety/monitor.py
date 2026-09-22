"""
T6.3 — Safety state monitor and fault injection helpers.
State machine: NORMAL -> DEGRADED -> SAFE (with hysteresis on recovery).
"""
from __future__ import annotations

import logging
import time

from yaazhi.config import settings
from yaazhi.types import SystemState

logger = logging.getLogger(__name__)


class SafetyMonitor:
    """
    Observes pipeline health and outputs current SystemState.
    NORMAL    -> DEGRADED: frame or IMU is stale beyond threshold
    DEGRADED  -> SAFE:     stale beyond safe_after_ms
    SAFE      -> NORMAL:   N consecutive good samples (hysteresis)
    """

    def __init__(self) -> None:
        sf = settings.safety
        self._frame_stale_ms = sf.frame_stale_ms
        self._imu_stale_ms = sf.imu_stale_ms
        self._safe_after_ms = sf.safe_after_ms
        self._recovery_n = sf.get("recovery_good_samples", 10)

        self._state = SystemState.NORMAL
        self._degraded_since_ns: int | None = None
        self._good_count = 0

    @property
    def state(self) -> SystemState:
        return self._state

    def observe(
        self,
        *,
        t_frame_ns: int,
        t_imu_ns: int,
        now_ns: int | None = None,
        fps: float = 60.0,
        inference_failed: bool = False,
    ) -> SystemState:
        now = now_ns if now_ns is not None else time.monotonic_ns()

        frame_age_ms = (now - t_frame_ns) / 1e6
        imu_age_ms = (now - t_imu_ns) / 1e6

        is_bad = (
            frame_age_ms > self._frame_stale_ms
            or imu_age_ms > self._imu_stale_ms
            or fps < 10.0
            or inference_failed
        )

        if is_bad:
            self._good_count = 0
            if self._state == SystemState.NORMAL:
                self._state = SystemState.DEGRADED
                self._degraded_since_ns = now
                logger.warning(
                    f"msg=state_transition to=DEGRADED "
                    f"frame_age_ms={frame_age_ms:.0f} imu_age_ms={imu_age_ms:.0f}"
                )
            elif self._state == SystemState.DEGRADED:
                assert self._degraded_since_ns is not None
                degraded_ms = (now - self._degraded_since_ns) / 1e6
                if degraded_ms > self._safe_after_ms:
                    self._state = SystemState.SAFE
                    logger.error(
                        f"msg=state_transition to=SAFE degraded_ms={degraded_ms:.0f}"
                    )
        else:
            if self._state == SystemState.SAFE:
                self._good_count += 1
                if self._good_count >= self._recovery_n:
                    self._state = SystemState.NORMAL
                    self._degraded_since_ns = None
                    self._good_count = 0
                    logger.info("msg=state_transition to=NORMAL recovered")
            else:
                self._state = SystemState.NORMAL
                self._degraded_since_ns = None
                self._good_count = 0

        return self._state
