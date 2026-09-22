"""
T3.3 — Reprojection quality evaluation.
Uses SyntheticThermalSource + SyntheticImuSource to measure
error with vs. without reprojection across all IMU profiles.
Outputs a report table to reports/reprojection_eval.md.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from yaazhi.ingestion.imu_synth import SyntheticImuSource, ImuProfile
from yaazhi.ingestion.sources import SyntheticThermalSource
from yaazhi.reprojection.camera import build_K
from yaazhi.reprojection.warp import reproject, rotation_delta

_DELAYS_MS = [0, 33, 66, 100, 150]
_OUT = Path("reports/reprojection_eval.md")


def _run_profile(profile: ImuProfile, delay_ms: float, n_frames: int = 50) -> dict:
    K = build_K()
    imu_src = SyntheticImuSource(profile=profile, rate_hz=200.0,
                                 noise_sigma_deg_per_s=0.0, seed=0)
    img_src = SyntheticThermalSource(fps=9.0, num_frames=n_frames)

    errors_no_reproj = []
    errors_reproj = []

    delay_ns = int(delay_ms * 1e6)

    for frame in img_src:
        t_c = frame.t_capture_ns
        t_d = t_c + delay_ns

        t_c_s = (t_c - imu_src._t0_ns) / 1e9
        t_d_s = (t_d - imu_src._t0_ns) / 1e9

        q_c = imu_src.ground_truth_at(t_c_s)
        q_d = imu_src.ground_truth_at(t_d_s)

        # Ground truth: frame shifted to orientation at t_d
        R_truth = rotation_delta(q_c, q_d)
        truth_result = reproject(frame.image, K, R_truth)
        mask = truth_result.valid_mask

        # No reprojection: use stale frame as-is
        stale = frame.image
        err_no = float(np.mean(np.abs(
            stale[mask].astype(np.float32) - truth_result.warped[mask].astype(np.float32)
        ))) if mask.any() else 0.0
        errors_no_reproj.append(err_no)

        # With reprojection: warp stale by R_delta
        warp_result = reproject(stale, K, R_truth)
        err_rp = float(np.mean(np.abs(
            warp_result.warped[mask].astype(np.float32) -
            truth_result.warped[mask].astype(np.float32)
        ))) if mask.any() else 0.0
        errors_reproj.append(err_rp)

    return {
        "profile": profile.value,
        "delay_ms": delay_ms,
        "err_no_reproj": float(np.mean(errors_no_reproj)),
        "err_reproj": float(np.mean(errors_reproj)),
        "benefit": float(np.mean(errors_no_reproj)) - float(np.mean(errors_reproj)),
    }


def run_eval(out_path: Path = _OUT) -> list[dict]:
    rows = []
    for profile in ImuProfile:
        for delay_ms in _DELAYS_MS:
            r = _run_profile(profile, delay_ms)
            rows.append(r)
            status = "OK" if r["benefit"] >= 0 else "REGRESSION"
            print(f"  {profile.value:20s}  delay={delay_ms:3.0f}ms  "
                  f"no_rp={r['err_no_reproj']:.2f}  rp={r['err_reproj']:.2f}  {status}")

    out_path.parent.mkdir(exist_ok=True)
    lines = [
        "# Reprojection Quality Evaluation",
        "",
        "Source: SyntheticThermalSource (animated gradient). IMU: SyntheticImuSource (no noise, ground truth).",
        "Metric: mean absolute pixel error vs. ground-truth orientation frame, measured on valid-mask region.",
        "Software-only — display hardware latency excluded.",
        "",
        "| Profile | Delay ms | MAE no-reproj | MAE reproj | Benefit | Pass? |",
        "|---------|----------|--------------|------------|---------|-------|",
    ]
    all_pass = True
    for r in rows:
        if r["profile"] == "still":
            pass_str = "N/A (still)"
        elif r["benefit"] >= 0:
            pass_str = "PASS"
        else:
            pass_str = "REGRESSION"
            all_pass = False

        lines.append(
            f"| {r['profile']} | {r['delay_ms']:.0f} | "
            f"{r['err_no_reproj']:.3f} | {r['err_reproj']:.3f} | "
            f"{r['benefit']:+.3f} | {pass_str} |"
        )

    lines += [
        "",
        f"**Overall result:** {'All moving profiles: reprojection <= no-reprojection PASS' if all_pass else 'Some regressions -- see REGRESSION rows'}",
        "",
        "> Note: error values reflect synthetic animated gradient; real thermal footage may differ.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {out_path}")
    return rows


if __name__ == "__main__":
    from yaazhi.logging_setup import setup_logging
    import logging
    setup_logging(logging.WARNING)
    print("Running reprojection evaluation...")
    run_eval()
