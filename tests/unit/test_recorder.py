# T1.5 — Recorder and replay tests
import tempfile
from pathlib import Path
import numpy as np
from yaazhi.types import ThermalFrame, ImuSample
from yaazhi.ingestion.recorder import SessionRecorder, ReplayFrameSource, ReplayImuSource


def _make_frame(seq: int, t_ns: int) -> ThermalFrame:
    return ThermalFrame(
        seq=seq,
        t_capture_ns=t_ns,
        image=np.full((120, 160), seq % 256, dtype=np.uint8)
    )


def _make_imu(t_ns: int) -> ImuSample:
    return ImuSample(t_ns=t_ns, quat_wxyz=(1.0, 0.0, 0.0, 0.0))


def test_record_and_replay_frame_count():
    with tempfile.TemporaryDirectory() as tmpdir:
        session_dir = Path(tmpdir) / "session"
        rec = SessionRecorder(session_dir, config_dict={"test": True})
        n = 25
        for i in range(n):
            rec.record_frame(_make_frame(i, i * 1000))
        rec.finalize()

        replay = ReplayFrameSource(session_dir)
        assert replay.total_frames == n


def test_record_and_replay_timestamps():
    with tempfile.TemporaryDirectory() as tmpdir:
        session_dir = Path(tmpdir) / "session"
        rec = SessionRecorder(session_dir, config_dict={})
        t_ns_list = [i * 111_111 for i in range(15)]
        for i, t in enumerate(t_ns_list):
            rec.record_frame(_make_frame(i, t))
        rec.finalize()

        replay = ReplayFrameSource(session_dir)
        replayed_ts = []
        while True:
            f = replay.read()
            if f is None:
                break
            replayed_ts.append(f.t_capture_ns)

        assert replayed_ts == t_ns_list


def test_imu_record_and_replay():
    with tempfile.TemporaryDirectory() as tmpdir:
        session_dir = Path(tmpdir) / "session"
        rec = SessionRecorder(session_dir, config_dict={})
        n = 30
        for i in range(n):
            rec.record_imu(_make_imu(i * 5_000_000))
        rec.finalize()

        replay_imu = ReplayImuSource(session_dir)
        assert replay_imu.total_samples == n
