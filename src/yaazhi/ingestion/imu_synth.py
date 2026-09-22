"""
T1.3 — Synthetic IMU generator.
Produces head orientation quaternions at 200 Hz from scripted motion profiles.
Deterministic with a seed. Provides ground truth at arbitrary time t.
"""
from __future__ import annotations

import logging
import math
import time
from enum import Enum
from typing import Callable

import numpy as np

from yaazhi.types import ImuSample

logger = logging.getLogger(__name__)


class ImuProfile(Enum):
    STILL = "still"
    SLOW_SCAN = "slow_scan"          # 30 deg/s
    FAST_TURN = "fast_turn"          # 200 deg/s smooth
    ABRUPT_REVERSAL = "abrupt_reversal"
    WALKING_BOB = "walking_bob"


def _axis_angle_to_quat(axis: np.ndarray, angle_rad: float) -> tuple[float, float, float, float]:
    """Convert axis-angle to quaternion (w, x, y, z)."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    s = math.sin(angle_rad / 2)
    return (math.cos(angle_rad / 2), axis[0] * s, axis[1] * s, axis[2] * s)


def _quat_mul(q1: tuple, q2: tuple) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _normalize_quat(q: tuple) -> tuple[float, float, float, float]:
    arr = np.array(q, dtype=np.float64)
    n = np.linalg.norm(arr)
    return tuple(arr / n)  # type: ignore[return-value]


# Profile functions: t (seconds) -> yaw angle (radians)
def _profile_still(t: float) -> float:
    return 0.0


def _profile_slow_scan(t: float) -> float:
    """30 deg/s sinusoidal sweep ±45 deg."""
    return math.radians(45) * math.sin(math.radians(30) * t)


def _profile_fast_turn(t: float) -> float:
    """200 deg/s smooth ramp, reverses at ±60 deg."""
    raw = math.radians(200) * t
    period = 2 * math.radians(60)
    # Triangle wave
    mod = raw % (2 * period)
    if mod < period:
        return mod - math.radians(60)
    return math.radians(60) - (mod - period)


def _profile_abrupt_reversal(t: float) -> float:
    """Alternates direction every 0.5 s at 200 deg/s."""
    half = 0.5
    cycles = int(t / half)
    phase = t - cycles * half
    direction = 1 if cycles % 2 == 0 else -1
    max_angle = math.radians(50)
    return min(max_angle, direction * math.radians(200) * phase) if direction > 0 else max(
        -max_angle, direction * math.radians(200) * phase
    )


def _profile_walking_bob(t: float) -> float:
    """Slow yaw drift + high-frequency bob for walking motion."""
    yaw = math.radians(20) * math.sin(math.radians(15) * t)
    bob = math.radians(5) * math.sin(2 * math.pi * 2 * t)  # 2 Hz bob
    return yaw + bob


_PROFILE_FNS: dict[ImuProfile, Callable[[float], float]] = {
    ImuProfile.STILL: _profile_still,
    ImuProfile.SLOW_SCAN: _profile_slow_scan,
    ImuProfile.FAST_TURN: _profile_fast_turn,
    ImuProfile.ABRUPT_REVERSAL: _profile_abrupt_reversal,
    ImuProfile.WALKING_BOB: _profile_walking_bob,
}

_YAW_AXIS = np.array([0.0, 1.0, 0.0])  # Y-up world convention


class SyntheticImuSource:
    """
    Generates head orientation quaternions at a configurable rate.
    All motion is modelled as pure yaw around the Y-axis for simplicity.
    Adds zero-mean Gaussian gyro noise (sigma_deg_per_s).

    Implements ImuSource protocol:
        def read(self) -> ImuSample | None
    """

    def __init__(
        self,
        profile: ImuProfile = ImuProfile.STILL,
        rate_hz: float = 200.0,
        noise_sigma_deg_per_s: float = 0.5,
        seed: int = 42,
        t0_ns: int | None = None,
    ) -> None:
        self._profile_fn = _PROFILE_FNS[profile]
        self._profile = profile
        self._rate_hz = rate_hz
        self._period_ns = int(1e9 / rate_hz)
        self._noise_sigma = math.radians(noise_sigma_deg_per_s) / rate_hz
        self._rng = np.random.default_rng(seed)
        self._t0_ns = t0_ns if t0_ns is not None else time.monotonic_ns()
        self._next_emit_ns = self._t0_ns

    # --- FrameSource-style read ---

    def read(self) -> ImuSample | None:
        now = time.monotonic_ns()
        wait_ns = self._next_emit_ns - now
        if wait_ns > 0:
            time.sleep(wait_ns / 1e9)

        t_ns = time.monotonic_ns()
        t_s = (t_ns - self._t0_ns) / 1e9
        quat = self.orientation_at(t_s)
        sample = ImuSample(t_ns=t_ns, quat_wxyz=quat)
        self._next_emit_ns += self._period_ns
        return sample

    # --- Ground truth (pure, deterministic, no noise) ---

    def ground_truth_at(self, t_s: float) -> tuple[float, float, float, float]:
        """True orientation quaternion at elapsed time t_s (no noise)."""
        yaw = self._profile_fn(t_s)
        return _normalize_quat(_axis_angle_to_quat(_YAW_AXIS, yaw))

    def orientation_at(self, t_s: float) -> tuple[float, float, float, float]:
        """Noisy quaternion at elapsed time t_s."""
        q_true = self.ground_truth_at(t_s)
        noise_angle = float(self._rng.normal(0.0, self._noise_sigma))
        noise_axis = self._rng.standard_normal(3)
        noise_axis /= np.linalg.norm(noise_axis) + 1e-12
        q_noise = _axis_angle_to_quat(noise_axis, noise_angle)
        return _normalize_quat(_quat_mul(q_true, q_noise))

    def peak_angular_rate_deg_per_s(self, duration_s: float = 2.0, samples: int = 1000) -> float:
        """Estimate peak angular rate (deg/s) over a duration."""
        dt = duration_s / samples
        max_rate = 0.0
        for i in range(samples - 1):
            t1, t2 = i * dt, (i + 1) * dt
            yaw1 = self._profile_fn(t1)
            yaw2 = self._profile_fn(t2)
            rate = abs(yaw2 - yaw1) / dt
            if rate > max_rate:
                max_rate = rate
        return math.degrees(max_rate)
