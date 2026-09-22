# T1.4 — Validation tests
import numpy as np
import pytest
from yaazhi.types import ThermalFrame, ImuSample
from yaazhi.ingestion.validation import validate_frame, validate_imu, RetryingSupervisor


def _good_frame(seq=0, t_ns=1000):
    img = np.zeros((120, 160), dtype=np.uint8)
    return ThermalFrame(seq=seq, t_capture_ns=t_ns, image=img)


def test_validate_frame_ok():
    f = _good_frame()
    assert validate_frame(f, last_seq=-1) is not None


def test_validate_frame_wrong_shape():
    img = np.zeros((64, 64), dtype=np.uint8)
    f = ThermalFrame(seq=0, t_capture_ns=0, image=img)
    assert validate_frame(f, expected_shape=(120, 160)) is None


def test_validate_frame_non_monotonic():
    f = _good_frame(seq=3)
    assert validate_frame(f, last_seq=5) is None


def test_validate_imu_ok():
    s = ImuSample(t_ns=0, quat_wxyz=(1.0, 0.0, 0.0, 0.0))
    assert validate_imu(s) is not None


def test_validate_imu_bad_norm():
    # norm check is in __post_init__ so invalid ImuSample raises on construction
    with pytest.raises(ValueError):
        ImuSample(t_ns=0, quat_wxyz=(0.5, 0.0, 0.0, 0.0))


class _FaultySource:
    """Source that raises exactly once per instance when fault_count > 0."""
    def __init__(self, should_fail: bool):
        self._should_fail = should_fail
        self._raised = False

    def read(self):
        if self._should_fail and not self._raised:
            self._raised = True
            raise IOError("Simulated fault")
        return "ok"


def test_retrying_supervisor_recovers():
    """
    Factory produces 3 faulty sources (each raises once) then a good one.
    Supervisor must recover from each exception and ultimately return 'ok'.
    """
    fail_count = 3
    calls = [0]

    def factory():
        idx = calls[0]
        calls[0] += 1
        return _FaultySource(should_fail=(idx < fail_count))

    sup = RetryingSupervisor(factory, name="test_src")
    result = sup.read()
    assert result == "ok"
    assert sup.exception_count == fail_count
