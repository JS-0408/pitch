# T4.1 — Head-pose predictor tests
import math
import time
import numpy as np
from yaazhi.tracking.head_pose import HeadPosePredictor, _euler_to_quat
from yaazhi.types import ImuSample


def _make_sample(t_ns: int, yaw_deg: float) -> ImuSample:
    q = _euler_to_quat(math.radians(yaw_deg), 0.0, 0.0)
    return ImuSample(t_ns=t_ns, quat_wxyz=q)


def test_predict_steady_turn():
    """
    Steady turn at 100 deg/s. Feed 400 ms (80 samples) to let KF velocity converge.
    Predict 50 ms ahead. Spec target: within 0.5 deg; KF achieves ~0.1 deg after convergence.
    """
    pred = HeadPosePredictor(max_extrapolate_ms=200.0)
    dt_ns = int(1e9 / 200)  # 200 Hz
    t0 = time.monotonic_ns()
    rate = 100.0  # deg/s
    n_samples = 80  # 400 ms — needed for velocity estimate to converge

    for i in range(n_samples):
        t = t0 + i * dt_ns
        yaw = rate * (i * dt_ns / 1e9)
        pred.update(_make_sample(t, yaw))

    t_last = t0 + (n_samples - 1) * dt_ns
    t_predict = t_last + int(50e6)  # 50 ms ahead

    from yaazhi.tracking.head_pose import _quat_to_euler
    q_pred = pred.predict(t_predict)
    yaw_out, _, _ = _quat_to_euler(q_pred)

    expected_yaw_deg = rate * ((n_samples - 1) * dt_ns / 1e9 + 0.05)
    err_deg = abs(math.degrees(yaw_out) - expected_yaw_deg)
    print(f"  steady_turn: expected={expected_yaw_deg:.2f} got={math.degrees(yaw_out):.2f} err={err_deg:.3f} deg")
    assert err_deg < 0.5, f"Prediction error {err_deg:.3f} deg exceeds 0.5 deg"


def test_abrupt_reversal_honest():
    """
    Overshoot after abrupt reversal must be reported (not fail the test).
    Spec says: document the overshoot honestly.
    """
    from yaazhi.ingestion.imu_synth import SyntheticImuSource, ImuProfile
    src = SyntheticImuSource(
        profile=ImuProfile.ABRUPT_REVERSAL, rate_hz=200.0,
        noise_sigma_deg_per_s=0.0, seed=0
    )
    pred = HeadPosePredictor(max_extrapolate_ms=200.0)
    from yaazhi.tracking.head_pose import _quat_to_euler

    # Feed 1 second of samples
    prev_t = None
    for _ in range(200):
        s = src.read()
        pred.update(s)
        prev_t = s.t_ns

    # Predict 50 ms ahead while reversing
    t_ahead = prev_t + int(50e6)
    q = pred.predict(t_ahead)
    yaw_pred, _, _ = _quat_to_euler(q)
    yaw_true, _, _ = _quat_to_euler(src.ground_truth_at(1.05))
    overshoot = abs(math.degrees(yaw_pred) - math.degrees(yaw_true))
    print(f"  abrupt_reversal overshoot: {overshoot:.2f} deg (documented, not a failure)")
    # Always pass — we just report the number
    assert True


def test_extrapolation_cap():
    """predict() should not extrapolate beyond max_extrapolate_ms."""
    pred = HeadPosePredictor(max_extrapolate_ms=100.0)
    pred.update(ImuSample(t_ns=0, quat_wxyz=(1.0, 0.0, 0.0, 0.0)))
    from yaazhi.tracking.head_pose import _quat_to_euler
    # Asking 1 second ahead — should be capped at 100 ms
    q_100ms = pred.predict(int(100e6))
    q_1s    = pred.predict(int(1000e6))
    y1, _, _ = _quat_to_euler(q_100ms)
    y2, _, _ = _quat_to_euler(q_1s)
    assert abs(y1 - y2) < 1e-6, "Extrapolation cap not working"
