"""
T1.1 — Canonical data types for the Yaazhi pipeline.
All inter-module data flows use these classes.  Nothing else.
Section 4.2 of the spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThermalFrame:
    seq: int
    t_capture_ns: int
    image: np.ndarray          # HxW uint8 display image
    raw: np.ndarray | None = None  # HxW uint16 original if available

    def __post_init__(self) -> None:
        if self.image.ndim != 2:
            raise ValueError(f"ThermalFrame.image must be 2-D (HxW), got shape {self.image.shape}")
        if self.image.dtype != np.uint8:
            raise ValueError(f"ThermalFrame.image must be uint8, got {self.image.dtype}")


@dataclass(frozen=True)
class ImuSample:
    t_ns: int
    quat_wxyz: tuple[float, float, float, float]  # head orientation, world frame

    def __post_init__(self) -> None:
        norm = float(np.linalg.norm(self.quat_wxyz))
        if not (0.99 <= norm <= 1.01):
            raise ValueError(f"ImuSample quaternion norm {norm:.6f} out of [0.99, 1.01]")


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    conf: float
    cls: int  # 0 = person

    def __post_init__(self) -> None:
        if not (0.0 <= self.conf <= 1.0):
            raise ValueError(f"Detection.conf {self.conf} out of [0, 1]")


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

class TrackStatus(Enum):
    TENTATIVE = 1
    CONFIRMED = 2
    COASTING = 3   # measurement missing, predicting
    LOST = 4


@dataclass
class Track:
    id: int
    status: TrackStatus
    cx: float
    cy: float
    w: float
    h: float
    vx: float
    vy: float
    conf: float           # 0..1, decays while COASTING
    t_last_update_ns: int


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Marker:
    track_id: int
    x: float
    y: float
    w: float
    h: float
    alpha: float   # 0..1 from confidence
    label: str


# ---------------------------------------------------------------------------
# System state
# ---------------------------------------------------------------------------

class SystemState(Enum):
    NORMAL = 1
    DEGRADED = 2    # stale thermal or IMU gaps: markers faded
    SAFE = 3        # sensors lost: overlay off, "CLEAR VIEW" banner
