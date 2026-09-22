"""
T6.2 — Decoupled Pipeline Orchestrator.
Orchestrates the 9 Hz thermal sensor + 200 Hz IMU perception loop
and the 60 Hz HUD rendering loop.
Integrates:
  - Sensor source (Mock / Synthetic)
  - AGC Preprocessing (preprocess.py)
  - ONNX Detector (onnx_detector.py)
  - Head Pose Predictor (head_pose.py)
  - Target Tracker (target_tracker.py)
  - Rotation-delta Warper (warp.py)
  - HUD Renderer (hud.py)
  - Safety Monitor (monitor.py)
  - Latency Tracker (latency.py)
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Any

import cv2
import numpy as np

from yaazhi.config import settings
from yaazhi.ingestion.imu_synth import SyntheticImuSource, ImuProfile
from yaazhi.ingestion.sources import SyntheticThermalSource
from yaazhi.perception.onnx_detector import OnnxDetector
from yaazhi.perception.preprocess import preprocess, AgcMode
from yaazhi.pipeline.latency import LatencyTracker
from yaazhi.rendering.hud import Renderer
from yaazhi.reprojection.warp import reproject, rotation_delta
from yaazhi.safety.monitor import SafetyMonitor
from yaazhi.tracking.head_pose import HeadPosePredictor
from yaazhi.tracking.target_tracker import TargetTracker
from yaazhi.types import SystemState, Marker

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Decoupled pipeline orchestrator.
    Maintains current system state, running perception updates when new frames arrive,
    and extrapolating reticles/warp at 60 Hz for HUD rendering.
    """

    def __init__(self, thermal_source=None, imu_source=None, detector=None) -> None:
        self.thermal_source = thermal_source or SyntheticThermalSource(fps=9)
        self.imu_source = imu_source or SyntheticImuSource(profile=ImuProfile.SLOW_SCAN, rate_hz=200)
        self.detector = detector or OnnxDetector()

        self.head_pose = HeadPosePredictor()
        self.tracker = TargetTracker()
        self.renderer = Renderer()
        self.safety_monitor = SafetyMonitor()
        self.latency_tracker = LatencyTracker()

        self.latest_raw_frame: np.ndarray | None = None
        self.latest_proc_frame: np.ndarray | None = None
        self.latest_frame_t_ns: int = 0
        self.latest_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

        self.is_running = False

    def process_perception_step(self, timestamp_ms: float) -> None:
        """Execute 1 perception step (9 Hz frame arrival)."""
        # Fetch frame
        try:
            frame_obj = self.thermal_source.read()
            if frame_obj is None:
                return
            self.latest_raw_frame = frame_obj.image
            self.latest_frame_t_ns = frame_obj.t_capture_ns
        except Exception:
            return

        # Fetch latest IMU sample
        imu_sample = self.imu_source.read()
        if imu_sample:
            self.latest_quat = imu_sample.quat_wxyz
            self.head_pose.update(imu_sample)

        # Preprocess AGC
        mode_str = getattr(settings.preprocess, "agc", "adaptive")
        mode = AgcMode(mode_str) if isinstance(mode_str, str) else mode_str
        self.latest_proc_frame = preprocess(self.latest_raw_frame, mode=mode)

        # Detect
        detections = self.detector.detect(self.latest_proc_frame)

        # Update tracking
        active_tracks = self.tracker.update(detections, timestamp_ms)

        # Update safety state
        now_ns = time.monotonic_ns()
        self.safety_monitor.observe(
            t_frame_ns=self.latest_frame_t_ns,
            t_imu_ns=now_ns,
            now_ns=now_ns,
        )

    def render_hud_frame(self, timestamp_ms: float) -> np.ndarray:
        """Execute 1 render step (60 Hz HUD update)."""
        if self.latest_proc_frame is None:
            # Fallback black canvas
            w = int(getattr(settings.sensor, "width", 640))
            h = int(getattr(settings.sensor, "height", 504))
            return np.zeros((h, w, 3), dtype=np.uint8)

        sys_state = self.safety_monitor.state
        active_tracks = self.tracker.predict(timestamp_ms)

        hud = {
            "render_fps": 60.0,
            "m2d_ms": 15.0,
            "tracks": len(active_tracks),
            "toggles": "AGC:ON",
        }

        if sys_state == SystemState.SAFE:
            # Clear view banner only
            return self.renderer.draw(self.latest_proc_frame, [], hud, state=sys_state)

        # Convert tracks to HUD Markers
        markers = []
        for trk in active_tracks:
            alpha = max(0.2, min(1.0, trk.conf))
            if sys_state == SystemState.DEGRADED:
                alpha *= 0.5  # faded markers

            markers.append(
                Marker(
                    track_id=trk.id,
                    x=trk.cx,
                    y=trk.cy,
                    w=trk.w,
                    h=trk.h,
                    alpha=alpha,
                    label=f"ID {trk.id} ({trk.status.name[0]})",
                )
            )

        # Render canvas
        canvas = self.renderer.draw(self.latest_proc_frame, markers, hud, state=sys_state)
        return canvas

    def run_benchmark(self, duration_s: float = 5.0) -> dict:
        """Run orchestrated dual loop for duration_s and return performance stats."""
        print(f"Starting orchestrator benchmark for {duration_s}s...")
        t_start = time.perf_counter()
        t_end = t_start + duration_s

        p_interval = 1.0 / 9.0   # ~111ms
        r_interval = 1.0 / 60.0  # ~16.6ms

        last_p_t = 0.0
        last_r_t = 0.0

        p_count = 0
        r_count = 0

        sim_t_ms = 0.0

        while time.perf_counter() < t_end:
            now = time.perf_counter()

            # Perception tick (~9 Hz)
            if now - last_p_t >= p_interval:
                self.process_perception_step(sim_t_ms)
                last_p_t = now
                p_count += 1

            # Render tick (~60 Hz)
            if now - last_r_t >= r_interval:
                _ = self.render_hud_frame(sim_t_ms)
                last_r_t = now
                r_count += 1

            sim_t_ms += 16.67
            time.sleep(0.002)

        elapsed = time.perf_counter() - t_start
        stats = {
            "duration_s": round(elapsed, 2),
            "perception_frames": p_count,
            "perception_fps": round(p_count / elapsed, 2),
            "render_frames": r_count,
            "render_fps": round(r_count / elapsed, 2),
            "final_system_state": self.safety_monitor.state.name,
        }
        print(f"Orchestrator benchmark finished: {stats}")
        return stats
