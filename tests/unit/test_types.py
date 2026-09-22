# T1.1 — Types unit tests
import pytest
import numpy as np
from yaazhi.types import (
    ThermalFrame, ImuSample, Detection,
    Track, TrackStatus, Marker, SystemState
)


def test_thermal_frame_valid():
    img = np.zeros((120, 160), dtype=np.uint8)
    f = ThermalFrame(seq=0, t_capture_ns=1000, image=img)
    assert f.seq == 0
    assert f.raw is None


def test_thermal_frame_immutable():
    img = np.zeros((120, 160), dtype=np.uint8)
    f = ThermalFrame(seq=0, t_capture_ns=1000, image=img)
    with pytest.raises((AttributeError, TypeError)):
        f.seq = 99  # type: ignore


def test_thermal_frame_wrong_shape():
    img = np.zeros((3, 120, 160), dtype=np.uint8)
    with pytest.raises(ValueError, match="2-D"):
        ThermalFrame(seq=0, t_capture_ns=0, image=img)


def test_thermal_frame_wrong_dtype():
    img = np.zeros((120, 160), dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        ThermalFrame(seq=0, t_capture_ns=0, image=img)


def test_imu_sample_valid():
    s = ImuSample(t_ns=0, quat_wxyz=(1.0, 0.0, 0.0, 0.0))
    assert s.quat_wxyz[0] == 1.0


def test_imu_sample_immutable():
    s = ImuSample(t_ns=0, quat_wxyz=(1.0, 0.0, 0.0, 0.0))
    with pytest.raises((AttributeError, TypeError)):
        s.t_ns = 1  # type: ignore


def test_imu_sample_bad_norm():
    with pytest.raises(ValueError, match="norm"):
        ImuSample(t_ns=0, quat_wxyz=(0.0, 0.0, 0.0, 0.0))


def test_detection_valid():
    d = Detection(xyxy=(0, 0, 10, 10), conf=0.9, cls=0)
    assert d.conf == 0.9


def test_detection_bad_conf():
    with pytest.raises(ValueError, match="conf"):
        Detection(xyxy=(0, 0, 10, 10), conf=1.5, cls=0)


def test_track_status_enum():
    assert TrackStatus.COASTING.value == 3


def test_system_state_enum():
    assert SystemState.SAFE.value == 3


def test_marker_immutable():
    m = Marker(track_id=1, x=10, y=20, w=30, h=40, alpha=0.8, label="P1")
    with pytest.raises((AttributeError, TypeError)):
        m.track_id = 99  # type: ignore
