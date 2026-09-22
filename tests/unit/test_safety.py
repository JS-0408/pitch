# T6.3 — Safety monitor tests
import time
from yaazhi.safety.monitor import SafetyMonitor
from yaazhi.types import SystemState


def _now():
    return time.monotonic_ns()


def test_normal_state_on_good_inputs():
    mon = SafetyMonitor()
    now = _now()
    state = mon.observe(t_frame_ns=now, t_imu_ns=now, now_ns=now, fps=60.0)
    assert state == SystemState.NORMAL


def test_degraded_on_stale_frame():
    mon = SafetyMonitor()
    now = _now()
    stale_frame_ns = now - int(500e6)  # 500 ms ago (stale_ms=400)
    state = mon.observe(t_frame_ns=stale_frame_ns, t_imu_ns=now, now_ns=now, fps=60.0)
    assert state == SystemState.DEGRADED


def test_safe_after_prolonged_stale():
    mon = SafetyMonitor()
    now = _now()
    stale = now - int(2000e6)   # 2000 ms = way past safe_after_ms=1200
    # First call: DEGRADED (entering degraded_since)
    mon.observe(t_frame_ns=stale, t_imu_ns=now, now_ns=now - int(1300e6), fps=60.0)
    # Second call: still bad, enough time passed
    state = mon.observe(t_frame_ns=stale, t_imu_ns=now, now_ns=now, fps=60.0)
    assert state == SystemState.SAFE


def test_recovery_requires_n_good_samples():
    mon = SafetyMonitor()
    now = _now()
    stale = now - int(2000e6)
    # Force into SAFE
    mon.observe(t_frame_ns=stale, t_imu_ns=now, now_ns=now - int(1300e6), fps=60.0)
    mon.observe(t_frame_ns=stale, t_imu_ns=now, now_ns=now, fps=60.0)
    assert mon.state == SystemState.SAFE

    # Feed good samples — need recovery_good_samples (10) before NORMAL
    good_ns = _now()
    for i in range(9):
        state = mon.observe(t_frame_ns=good_ns, t_imu_ns=good_ns, now_ns=good_ns, fps=60.0)
        assert state == SystemState.SAFE  # not yet recovered

    state = mon.observe(t_frame_ns=good_ns, t_imu_ns=good_ns, now_ns=good_ns, fps=60.0)
    assert state == SystemState.NORMAL
