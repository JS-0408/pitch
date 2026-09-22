# T1.3 — Synthetic IMU generator tests
import math
import numpy as np
from yaazhi.ingestion.imu_synth import SyntheticImuSource, ImuProfile


def test_quaternion_unit_norm():
    src = SyntheticImuSource(profile=ImuProfile.FAST_TURN, seed=0, rate_hz=200)
    for _ in range(50):
        s = src.read()
        assert s is not None
        norm = math.sqrt(sum(x*x for x in s.quat_wxyz))
        assert abs(norm - 1.0) < 0.01, f"Norm {norm:.6f} not unit"


def test_still_profile_near_zero_rate():
    src = SyntheticImuSource(profile=ImuProfile.STILL, seed=0, noise_sigma_deg_per_s=0.0)
    rate = src.peak_angular_rate_deg_per_s(duration_s=1.0)
    assert rate < 1.0, f"Still profile has rate {rate:.2f} deg/s"


def test_slow_scan_rate():
    src = SyntheticImuSource(profile=ImuProfile.SLOW_SCAN, seed=0, noise_sigma_deg_per_s=0.0)
    rate = src.peak_angular_rate_deg_per_s(duration_s=5.0)
    # Sinusoidal: peak rate = 30 deg/s (angular freq * amplitude = 30 * π/180 * 45 → ~rad/s check)
    assert rate <= 35.0, f"Slow_scan peak rate {rate:.1f} deg/s exceeds expected"


def test_fast_turn_peak_rate():
    src = SyntheticImuSource(profile=ImuProfile.FAST_TURN, seed=0, noise_sigma_deg_per_s=0.0)
    rate = src.peak_angular_rate_deg_per_s(duration_s=3.0)
    assert rate >= 100.0, f"Fast_turn peak rate {rate:.1f} deg/s too low"


def test_ground_truth_deterministic():
    src = SyntheticImuSource(profile=ImuProfile.WALKING_BOB, seed=42)
    q1 = src.ground_truth_at(1.0)
    q2 = src.ground_truth_at(1.0)
    assert q1 == q2


def test_all_profiles_unit_norm():
    for profile in ImuProfile:
        src = SyntheticImuSource(profile=profile, seed=7, rate_hz=200)
        s = src.read()
        assert s is not None
        norm = math.sqrt(sum(x*x for x in s.quat_wxyz))
        assert abs(norm - 1.0) < 0.01, f"Profile {profile} norm {norm:.6f}"
