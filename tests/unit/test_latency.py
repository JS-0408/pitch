# T3.4 — Latency tracker tests
import time
from yaazhi.pipeline.latency import LatencyTracker, FrameTimings


def test_latency_stats_empty():
    lt = LatencyTracker()
    assert lt.stats() is None


def test_latency_stats_basic():
    lt = LatencyTracker()
    now = time.monotonic_ns()
    for i in range(50):
        t_capture = now + i * 111_111_111     # 111 ms per frame
        t_imu     = t_capture + 5_000_000    # 5 ms later
        t_display = t_capture + 20_000_000   # 20 ms after capture
        lt.record(FrameTimings(t_capture, t_imu, t_display))

    s = lt.stats()
    assert s is not None
    assert s.n_samples == 50
    # m2d = t_display - t_imu = 15 ms
    assert abs(s.motion_to_display_median_ms - 15.0) < 0.5
    # frame age = t_display - t_capture = 20 ms
    assert abs(s.frame_age_median_ms - 20.0) < 0.5


def test_latency_rolling_window():
    lt = LatencyTracker(window=10)
    now = time.monotonic_ns()
    for i in range(20):
        lt.record(FrameTimings(now, now + 1_000_000, now + 10_000_000))
    s = lt.stats()
    assert s.n_samples == 10  # window capped
