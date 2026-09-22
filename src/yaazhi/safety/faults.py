"""
T6.3 — Fault Injectors for safety system testing.
Injects IMU drop, frame freeze, or detector latency spikes.
"""
from __future__ import annotations

import time
from enum import Enum


class FaultType(Enum):
    NONE = "none"
    IMU_DROP = "imu_drop"
    FRAME_FREEZE = "frame_freeze"
    DETECTOR_SPIKE = "detector_spike"


class FaultInjector:
    def __init__(self) -> None:
        self.active_fault = FaultType.NONE
        self._freeze_t_ns: int | None = None
        self._spike_delay_s: float = 0.5

    def set_fault(self, fault: FaultType) -> None:
        self.active_fault = fault
        if fault != FaultType.FRAME_FREEZE:
            self._freeze_t_ns = None

    def process_imu_timestamp(self, t_ns: int) -> int:
        if self.active_fault == FaultType.IMU_DROP:
            # Simulate IMU stopped updating 5 seconds ago
            return t_ns - 5_000_000_000
        return t_ns

    def process_frame_timestamp(self, t_ns: int) -> int:
        if self.active_fault == FaultType.FRAME_FREEZE:
            if self._freeze_t_ns is None:
                self._freeze_t_ns = t_ns
            return self._freeze_t_ns
        return t_ns

    def apply_detector_delay(self) -> None:
        if self.active_fault == FaultType.DETECTOR_SPIKE:
            time.sleep(self._spike_delay_s)
