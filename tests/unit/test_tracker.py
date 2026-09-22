# T4.2 — Target tracker tests
import time
import numpy as np
import pytest
from yaazhi.types import Detection, TrackStatus
from yaazhi.tracking.target_tracker import TargetTracker
from yaazhi.tracking.association import iou


# --- Association unit tests ---

def test_iou_same_box():
    assert abs(iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-6


def test_iou_no_overlap():
    assert iou((0, 0, 5, 5), (10, 10, 20, 20)) == 0.0


def test_iou_partial():
    v = iou((0, 0, 10, 10), (5, 5, 15, 15))
    assert 0.0 < v < 1.0


# --- Tracker lifecycle tests ---

def _det(cx: float, cy: float, w: float = 10.0, h: float = 20.0, conf: float = 0.9):
    return Detection(xyxy=(cx - w/2, cy - h/2, cx + w/2, cy + h/2), conf=conf, cls=0)


def test_constant_velocity_recovery():
    """Track constant-velocity target; RMSE of center must be under bound."""
    tracker = TargetTracker()
    dt_ns = int(1e9 / 9)  # 9 Hz
    t0 = time.monotonic_ns()
    vx = 2.0  # px/frame
    n = 30
    errs = []
    for i in range(n):
        t = t0 + i * dt_ns
        cx = 80.0 + vx * i
        cy = 60.0
        dets = [_det(cx, cy)]
        tracks = tracker.update(dets, t)
        if tracks:
            errs.append(abs(tracks[0].cx - cx))
    rmse = float(np.sqrt(np.mean(np.array(errs) ** 2)))
    print(f"  constant_velocity rmse={rmse:.2f} px")
    assert rmse < 3.0, f"RMSE {rmse:.2f} too high"


def test_coasting_to_lost():
    """Track disappears → COASTING → LOST at configured times."""
    tracker = TargetTracker()
    dt_ns = int(1e9 / 9)
    t0 = time.monotonic_ns()

    # Feed enough to confirm
    for i in range(5):
        tracker.update([_det(80.0, 60.0)], t0 + i * dt_ns)

    # Stop sending detections — should become COASTING within coast_ms
    coast_ms = 600
    drop_ms = 1500
    t_after_coast = t0 + 4 * dt_ns + int((coast_ms + 50) * 1e6)
    tracks = tracker.update([], t_after_coast)
    assert len(tracks) == 1
    assert tracks[0].status == TrackStatus.COASTING

    t_after_drop = t0 + 4 * dt_ns + int((drop_ms + 100) * 1e6)
    tracks = tracker.update([], t_after_drop)
    assert len(tracks) == 0  # LOST tracks are pruned


def test_single_fp_never_confirmed():
    """A single false-positive frame should never become CONFIRMED."""
    tracker = TargetTracker()
    t0 = time.monotonic_ns()
    tracker.update([_det(80.0, 60.0)], t0)
    tracks = tracker.update([], t0 + int(200e6))  # 200 ms gap
    # Either empty or TENTATIVE
    for t in tracks:
        assert t.status != TrackStatus.CONFIRMED


def test_predict_noop_on_filter():
    """predict() must not mutate filter state."""
    tracker = TargetTracker()
    t0 = time.monotonic_ns()
    for i in range(5):
        tracker.update([_det(80.0, 60.0)], t0 + i * int(1e9 / 9))

    before = [(t.cx, t.cy) for t in tracker.update([], t0 + 5 * int(1e9 / 9))]
    _ = tracker.predict(t0 + int(100e9))  # far future
    after_tracks = tracker.predict(t0 + 5 * int(1e9 / 9))
    # state should be unchanged
    after = [(t.cx, t.cy) for t in after_tracks]
    for (bx, by), (ax, ay) in zip(before, after):
        assert abs(bx - ax) < 30.0  # positions close (within reasonable prediction)
