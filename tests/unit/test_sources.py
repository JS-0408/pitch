# T1.2 — SyntheticThermalSource rate test (fast, no real data needed)
import time
import numpy as np
from yaazhi.ingestion.sources import SyntheticThermalSource


def test_synthetic_source_emits_frames():
    src = SyntheticThermalSource(fps=9.0, num_frames=9)
    frames = []
    while True:
        f = src.read()
        if f is None:
            break
        frames.append(f)
    assert len(frames) == 9


def test_synthetic_source_shape():
    src = SyntheticThermalSource(sensor_h=120, sensor_w=160, fps=9.0, num_frames=1)
    f = src.read()
    assert f is not None
    assert f.image.shape == (120, 160)
    assert f.image.dtype == np.uint8


def test_synthetic_source_monotonic_seq():
    src = SyntheticThermalSource(fps=100.0, num_frames=10)
    seqs = []
    while True:
        f = src.read()
        if f is None:
            break
        seqs.append(f.seq)
    assert seqs == list(range(10))


def test_synthetic_source_approx_rate():
    fps_target = 30.0
    n = 30
    src = SyntheticThermalSource(fps=fps_target, num_frames=n)
    t0 = time.monotonic()
    while src.read() is not None:
        pass
    elapsed = time.monotonic() - t0
    measured = n / elapsed
    assert abs(measured - fps_target) / fps_target < 0.05, (
        f"Rate {measured:.1f} too far from target {fps_target}"
    )
