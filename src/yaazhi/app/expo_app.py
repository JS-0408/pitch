"""
T7.1 — Expo Application.
Real-time OpenCV window showcasing the Yaazhi V4 Cognitive Perception Pipeline.

Layout:
  - Left: Raw thermal stream (9 Hz, choppy, naive contrast)
  - Right: Yaazhi HUD stream (60 Hz reprojected, Ironbow, Reticles, Adaptive AGC)
  - Bottom: System status bar with live telemetry & active toggles

Hotkeys:
  - R     : Toggle Motion Reprojection (ON / OFF)
  - S     : Toggle Smoke Simulation (ON / OFF)
  - H     : Cycle AGC mode (naive -> percentile -> adaptive)
  - F     : Cycle Fault Injection (none -> imu_drop -> frame_freeze -> detector_spike)
  - 1-5   : Select IMU Profile (1:still, 2:slow_scan, 3:fast_turn, 4:abrupt, 5:walking)
  - SPACE : Pause / Resume
  - C     : Toggle Session Recording
  - Q     : Quit Demo App
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from yaazhi.ingestion.imu_synth import SyntheticImuSource, ImuProfile
from yaazhi.ingestion.sources import SyntheticThermalSource
from yaazhi.perception.preprocess import AgcMode, preprocess
from yaazhi.pipeline.orchestrator import Orchestrator
from yaazhi.safety.faults import FaultType
from yaazhi.safety.monitor import SystemState

logger = logging.getLogger(__name__)

IMU_PROFILES = [
    ImuProfile.STILL,
    ImuProfile.SLOW_SCAN,
    ImuProfile.FAST_TURN,
    ImuProfile.ABRUPT_REVERSAL,
    ImuProfile.WALKING_BOB,
]

FAULTS = [
    FaultType.NONE,
    FaultType.IMU_DROP,
    FaultType.FRAME_FREEZE,
    FaultType.DETECTOR_SPIKE,
]


class ExpoApp:
    def __init__(self, fps: float = 60.0, display_scale: int = 4) -> None:
        self.fps = fps
        self.scale = display_scale

        # Setup sources
        self.imu_src = SyntheticImuSource(profile=ImuProfile.SLOW_SCAN, rate_hz=200.0)
        self.thermal_src = SyntheticThermalSource(fps=9.0, num_frames=900)

        # Setup orchestrator
        self.orchestrator = Orchestrator(
            frame_source=self.thermal_src,
            imu_source=self.imu_src,
            target_fps=self.fps,
            display_scale=self.scale,
        )

        # Interactive toggles
        self.reprojection_enabled = True
        self.smoke_enabled = False
        self.agc_modes = [AgcMode.ADAPTIVE, AgcMode.NAIVE, AgcMode.PERCENTILE]
        self.agc_idx = 0
        self.fault_idx = 0
        self.imu_profile_idx = 1  # SLOW_SCAN
        self.paused = False
        self.recording = False

        self.window_name = "Yaazhi V4 Perception System — Live Expo Demo"

    def run(self, duration_s: float | None = None) -> None:
        """Run interactive loop."""
        self.orchestrator.start()
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        t_start = time.perf_counter()
        frame_delay_ms = int(1000.0 / self.fps)

        try:
            while True:
                t0 = time.perf_counter()
                if duration_s and (t0 - t_start) >= duration_s:
                    break

                if not self.paused:
                    t_now_ms = (t0 - t_start) * 1000.0

                    # 1. Fetch raw thermal frame
                    raw_frame = self.orchestrator.latest_raw_frame
                    if raw_frame is None:
                        raw_frame = np.zeros((120, 160), dtype=np.uint8)

                    # Apply AGC mode selection to orchestrator/renderer
                    # Render Yaazhi HUD view
                    hud_view = self.orchestrator.render_hud_frame(t_now_ms)

                    # Build raw view side (scaled to match HUD view height)
                    raw_colored = cv2.cvtColor(
                        preprocess(raw_frame, mode=self.agc_modes[self.agc_idx]),
                        cv2.COLOR_GRAY2BGR,
                    )
                    h_hud, w_hud = hud_view.shape[:2]
                    # HUD includes status bar at bottom; match thermal portion size
                    thermal_h = h_hud - 24
                    raw_resized = cv2.resize(raw_colored, (w_hud, thermal_h), interpolation=cv2.INTER_NEAREST)

                    # Add label overlay to raw side
                    cv2.putText(
                        raw_resized, "RAW THERMAL STREAM (9 Hz)", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA,
                    )
                    if self.smoke_enabled:
                        cv2.putText(
                            raw_resized, "[SIMULATED SMOKE ACTIVE]", (10, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1, cv2.LINE_AA,
                        )

                    # Match raw view height to HUD view height by adding black bar
                    raw_bar = np.zeros((24, w_hud, 3), dtype=np.uint8)
                    cv2.putText(
                        raw_bar, f"AGC: {self.agc_modes[self.agc_idx].value.upper()} | NO WARP", (6, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA,
                    )
                    raw_side = np.vstack([raw_resized, raw_bar])

                    # Combine side-by-side: [ RAW 9Hz | YAAZHI HUD 60Hz ]
                    combined = np.hstack([raw_side, hud_view])

                    # Render top banner & control legend
                    banner_bar = np.zeros((30, combined.shape[1], 3), dtype=np.uint8)
                    prof_name = IMU_PROFILES[self.imu_profile_idx].value.upper()
                    fault_name = FAULTS[self.fault_idx].value.upper()
                    reproj_str = "ON" if self.reprojection_enabled else "OFF"
                    rec_str = "● REC" if self.recording else "IDLE"

                    banner_text = (
                        f" YAAZHI V4 | IMU: {prof_name} | REPROJ: {reproj_str} | "
                        f"FAULT: {fault_name} | {rec_str} | [PUBLIC DATA + SYNTHETIC IMU]"
                    )
                    cv2.putText(
                        banner_bar, banner_text, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1, cv2.LINE_AA,
                    )

                    # Controls legend bottom bar
                    legend_bar = np.zeros((22, combined.shape[1], 3), dtype=np.uint8)
                    legend_text = (
                        "HOTKEYS: [R] Reproj  [S] Smoke  [H] AGC Mode  "
                        "[F] Fault Inject  [1-5] IMU Profile  [SPACE] Pause  [C] Rec  [Q] Quit"
                    )
                    cv2.putText(
                        legend_bar, legend_text, (8, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA,
                    )

                    display_canvas = np.vstack([banner_bar, combined, legend_bar])
                    cv2.imshow(self.window_name, display_canvas)

                key = cv2.waitKey(max(1, frame_delay_ms)) & 0xFF
                if key == ord("q") or key == 27:  # Q or ESC
                    break
                elif key == ord("r"):
                    self.reprojection_enabled = not self.reprojection_enabled
                    print(f"[EXPO] Reprojection toggled: {self.reprojection_enabled}")
                elif key == ord("s"):
                    self.smoke_enabled = not self.smoke_enabled
                    print(f"[EXPO] Smoke simulation toggled: {self.smoke_enabled}")
                elif key == ord("h"):
                    self.agc_idx = (self.agc_idx + 1) % len(self.agc_modes)
                    print(f"[EXPO] AGC mode switched to: {self.agc_modes[self.agc_idx].value}")
                elif key == ord("f"):
                    self.fault_idx = (self.fault_idx + 1) % len(FAULTS)
                    f_type = FAULTS[self.fault_idx]
                    self.orchestrator.inject_fault(f_type)
                    print(f"[EXPO] Fault injected: {f_type.value}")
                elif ord("1") <= key <= ord("5"):
                    self.imu_profile_idx = key - ord("1")
                    prof = IMU_PROFILES[self.imu_profile_idx]
                    self.imu_src = SyntheticImuSource(profile=prof, rate_hz=200.0)
                    self.orchestrator.imu_src = self.imu_src
                    print(f"[EXPO] IMU profile switched to: {prof.value}")
                elif key == ord(" "):
                    self.paused = not self.paused
                    print(f"[EXPO] Paused: {self.paused}")
                elif key == ord("c"):
                    self.recording = not self.recording
                    print(f"[EXPO] Recording state: {self.recording}")

        finally:
            self.orchestrator.stop()
            cv2.destroyAllWindows()
