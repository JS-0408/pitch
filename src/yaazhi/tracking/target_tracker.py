"""
T4.2 — Multi-target tracker.
Constant-velocity Kalman per track (cx, cy, vx, vy; w, h smoothed).
Track lifecycle: TENTATIVE → CONFIRMED → COASTING → LOST.
predict(t_ns) extrapolates for the render loop WITHOUT mutating filter state.
"""
from __future__ import annotations

import logging
import time
from copy import deepcopy

import numpy as np

from yaazhi.config import settings
from yaazhi.types import Detection, Track, TrackStatus
from yaazhi.tracking.association import associate

logger = logging.getLogger(__name__)

# Kalman state: [cx, cy, vx, vy]
_DIM = 4


def _make_kalman(cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    """Initial state vector and covariance."""
    x = np.array([cx, cy, 0.0, 0.0], dtype=np.float64)
    P = np.diag([10.0, 10.0, 1000.0, 1000.0])
    return x, P


def _predict_kalman(x: np.ndarray, P: np.ndarray, dt: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1],
    ], dtype=np.float64)
    Q = np.diag([1.0, 1.0, 10.0, 10.0]) * max(dt, 0.001)
    x_p = F @ x
    P_p = F @ P @ F.T + Q
    return x_p, P_p


def _update_kalman(x_p: np.ndarray, P_p: np.ndarray,
                   cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float64)
    R = np.diag([5.0, 5.0])
    z = np.array([cx, cy], dtype=np.float64)
    y = z - H @ x_p
    S = H @ P_p @ H.T + R
    K = P_p @ H.T @ np.linalg.inv(S)
    x_u = x_p + K @ y
    P_u = (np.eye(_DIM) - K @ H) @ P_p
    return x_u, P_u


class _KalmanTrack:
    next_id = 1

    def __init__(self, det: Detection, t_ns: int, cfg) -> None:
        self.id = _KalmanTrack.next_id
        _KalmanTrack.next_id += 1
        cx = (det.xyxy[0] + det.xyxy[2]) / 2
        cy = (det.xyxy[1] + det.xyxy[3]) / 2
        self.w = det.xyxy[2] - det.xyxy[0]
        self.h = det.xyxy[3] - det.xyxy[1]
        self.x, self.P = _make_kalman(cx, cy)
        self.status = TrackStatus.TENTATIVE
        self.conf = det.conf
        self.hits = 1
        self.t_last_update_ns = t_ns
        self.t_last_seen_ns = t_ns
        self._confirm_hits = cfg.tracker.confirm_hits
        self._coast_ms = cfg.tracker.coast_ms
        self._drop_ms = cfg.tracker.drop_ms
        self._decay = cfg.tracker.conf_decay_per_s

    def predict_to(self, t_ns: int) -> None:
        """Advance filter state (mutates)."""
        dt = (t_ns - self.t_last_update_ns) / 1e9
        if dt > 0:
            self.x, self.P = _predict_kalman(self.x, self.P, dt)
            self.t_last_update_ns = t_ns

    def update(self, det: Detection, t_ns: int) -> None:
        cx = (det.xyxy[0] + det.xyxy[2]) / 2
        cy = (det.xyxy[1] + det.xyxy[3]) / 2
        # Smooth w, h with EMA
        alpha = 0.3
        self.w = alpha * (det.xyxy[2] - det.xyxy[0]) + (1 - alpha) * self.w
        self.h = alpha * (det.xyxy[3] - det.xyxy[1]) + (1 - alpha) * self.h

        dt = (t_ns - self.t_last_update_ns) / 1e9
        x_p, P_p = _predict_kalman(self.x, self.P, dt)
        self.x, self.P = _update_kalman(x_p, P_p, cx, cy)
        self.t_last_update_ns = t_ns
        self.t_last_seen_ns = t_ns
        self.conf = det.conf
        self.hits += 1
        if self.status == TrackStatus.TENTATIVE and self.hits >= self._confirm_hits:
            self.status = TrackStatus.CONFIRMED
        elif self.status == TrackStatus.COASTING:
            self.status = TrackStatus.CONFIRMED

    def mark_missed(self, now_ns: int) -> None:
        """Called when no detection matched this track."""
        dt = (now_ns - self.t_last_seen_ns) / 1e9
        self.conf = max(0.0, self.conf - self._decay * dt)

        coast_ns = self._coast_ms * 1e6
        drop_ns = self._drop_ms * 1e6
        silent_ns = now_ns - self.t_last_seen_ns

        if silent_ns > drop_ns:
            self.status = TrackStatus.LOST
        elif silent_ns > coast_ns:
            self.status = TrackStatus.COASTING
        elif self.status == TrackStatus.CONFIRMED:
            self.status = TrackStatus.COASTING

    def to_track(self) -> Track:
        return Track(
            id=self.id,
            status=self.status,
            cx=float(self.x[0]),
            cy=float(self.x[1]),
            w=self.w,
            h=self.h,
            vx=float(self.x[2]),
            vy=float(self.x[3]),
            conf=self.conf,
            t_last_update_ns=self.t_last_update_ns,
        )

    def snapshot_predict(self, t_ns: int) -> Track:
        """Return predicted Track at t_ns WITHOUT mutating the filter."""
        dt = (t_ns - self.t_last_update_ns) / 1e9
        if dt > 0:
            x_p, _ = _predict_kalman(self.x, self.P, dt)
        else:
            x_p = self.x
        decay = self._decay * max(0.0, dt) if self.status == TrackStatus.COASTING else 0.0
        return Track(
            id=self.id,
            status=self.status,
            cx=float(x_p[0]),
            cy=float(x_p[1]),
            w=self.w, h=self.h,
            vx=float(x_p[2]),
            vy=float(x_p[3]),
            conf=max(0.0, self.conf - decay),
            t_last_update_ns=self.t_last_update_ns,
        )


class TargetTracker:
    """
    Manages a pool of KalmanTracks.
    update() ingests detections; returns active tracks.
    predict() extrapolates for the render loop (read-only, no state mutation).
    """

    def __init__(self) -> None:
        self._tracks: list[_KalmanTrack] = []
        self._cfg = settings

    def update(self, dets: list[Detection], t_ns: int) -> list[Track]:
        gate_iou = self._cfg.tracker.gate_iou

        # Predict all tracks to current time
        for t in self._tracks:
            t.predict_to(t_ns)

        active = [t for t in self._tracks if t.status != TrackStatus.LOST]
        matches, unmatched_tracks, unmatched_dets = associate(
            [t.to_track() for t in active], dets, gate_iou=gate_iou
        )

        # Update matched
        for ti, di in matches:
            active[ti].update(dets[di], t_ns)

        # Mark unmatched tracks as missed
        for ti in unmatched_tracks:
            active[ti].mark_missed(t_ns)

        # Spawn new tracks for unmatched detections
        for di in unmatched_dets:
            self._tracks.append(_KalmanTrack(dets[di], t_ns, self._cfg))

        # Prune LOST tracks
        self._tracks = [t for t in self._tracks if t.status != TrackStatus.LOST]

        return [t.to_track() for t in self._tracks]

    def predict(self, t_ns: int) -> list[Track]:
        """Non-mutating prediction to t_ns for the render loop."""
        return [
            t.snapshot_predict(t_ns)
            for t in self._tracks
            if t.status != TrackStatus.LOST
        ]

    @property
    def track_count(self) -> int:
        return len([t for t in self._tracks if t.status != TrackStatus.LOST])
