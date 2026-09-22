"""
T7.1 — Demo runner script.
Launches the Yaazhi V4 interactive Expo presentation.

Usage:
    python scripts/run_demo.py [--scale 4] [--fps 60] [--headless]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src in sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from yaazhi.app.expo_app import ExpoApp
from yaazhi.logging_setup import setup_logging
import logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Yaazhi V4 Perception System — Expo Demo Runner")
    parser.add_argument("--scale", type=int, default=4, help="Display scaling factor (default: 4)")
    parser.add_argument("--fps", type=float, default=60.0, help="Target HUD render FPS (default: 60.0)")
    parser.add_argument("--duration", type=float, default=None, help="Auto-close after duration in seconds")
    args = parser.parse_args()

    setup_logging(logging.INFO)
    print("=" * 65)
    print(" YAAZHI V4 — COGNITIVE PERCEPTION PIPELINE DEMO")
    print("=" * 65)
    print(" Controls:")
    print("   [R]     Toggle Motion Reprojection (ON / OFF)")
    print("   [S]     Toggle Simulated Smoke")
    print("   [H]     Cycle AGC Mode (Adaptive -> Naive -> Percentile)")
    print("   [F]     Cycle Fault Injection (IMU drop, Freeze, Latency spike)")
    print("   [1-5]   Switch IMU Profile (1:still 2:slow 3:fast 4:abrupt 5:bob)")
    print("   [SPACE] Pause / Resume Demo")
    print("   [C]     Toggle Session Recording")
    print("   [Q]     Quit Demo")
    print("=" * 65)

    app = ExpoApp(fps=args.fps, display_scale=args.scale)
    app.run(duration_s=args.duration)


if __name__ == "__main__":
    main()
