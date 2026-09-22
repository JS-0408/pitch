"""
T1.5 — Session recorder and replay sources.
Session format: directory with frames.npy chunks, imu.csv, markers.jsonl, meta.json.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from yaazhi.types import ThermalFrame, ImuSample, Marker

logger = logging.getLogger(__name__)

_FRAMES_CHUNK_SIZE = 100   # frames per .npy chunk


def _git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class SessionRecorder:
    """
    Records a live session to a directory.
    Call record_frame(), record_imu(), record_marker() from the pipeline.
    Call finalize() when done.
    """

    def __init__(self, session_dir: Path, config_dict: dict) -> None:
        self._dir = Path(session_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._frames: list[np.ndarray] = []
        self._seq_list: list[int] = []
        self._t_list: list[int] = []
        self._chunk_idx = 0
        self._imu_rows: list[dict] = []
        self._marker_file = (self._dir / "markers.jsonl").open("w", encoding="utf-8")

        meta = {
            "config": config_dict,
            "git_hash": _git_hash(),
            "start_time_ns": time.monotonic_ns(),
        }
        with (self._dir / "meta.json").open("w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"msg=session_started path={self._dir}")

    def record_frame(self, frame: ThermalFrame) -> None:
        self._frames.append(frame.image.copy())
        self._seq_list.append(frame.seq)
        self._t_list.append(frame.t_capture_ns)
        if len(self._frames) >= _FRAMES_CHUNK_SIZE:
            self._flush_frames()

    def record_imu(self, sample: ImuSample) -> None:
        self._imu_rows.append({
            "t_ns": sample.t_ns,
            "w": sample.quat_wxyz[0],
            "x": sample.quat_wxyz[1],
            "y": sample.quat_wxyz[2],
            "z": sample.quat_wxyz[3],
        })

    def record_marker(self, marker: Marker, t_ns: int) -> None:
        row = {
            "t_ns": t_ns,
            "track_id": marker.track_id,
            "x": marker.x, "y": marker.y,
            "w": marker.w, "h": marker.h,
            "alpha": marker.alpha, "label": marker.label,
        }
        self._marker_file.write(json.dumps(row) + "\n")

    def _flush_frames(self) -> None:
        chunk_path = self._dir / f"frames_{self._chunk_idx:04d}.npy"
        arr = np.stack(self._frames, axis=0)
        np.save(str(chunk_path), arr)
        meta_path = self._dir / f"frames_{self._chunk_idx:04d}_meta.json"
        with meta_path.open("w") as f:
            json.dump({"seqs": self._seq_list, "t_ns": self._t_list}, f)
        self._chunk_idx += 1
        self._frames.clear()
        self._seq_list.clear()
        self._t_list.clear()

    def finalize(self) -> None:
        if self._frames:
            self._flush_frames()
        # Write IMU CSV
        imu_path = self._dir / "imu.csv"
        with imu_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["t_ns", "w", "x", "y", "z"])
            writer.writeheader()
            writer.writerows(self._imu_rows)
        self._marker_file.close()
        logger.info(f"msg=session_finalized path={self._dir} chunks={self._chunk_idx}")


# ---------------------------------------------------------------------------
# Replay sources
# ---------------------------------------------------------------------------

class ReplayFrameSource:
    """Replays a recorded session frame-by-frame (same protocol as MockThermalSource)."""

    def __init__(self, session_dir: Path) -> None:
        self._dir = Path(session_dir)
        self._frames: list[tuple[int, int, np.ndarray]] = []  # (seq, t_ns, image)
        self._idx = 0
        self._load()

    def _load(self) -> None:
        chunks = sorted(self._dir.glob("frames_*.npy"))
        for chunk in chunks:
            arr = np.load(str(chunk))
            meta_path = chunk.with_suffix("").with_name(chunk.stem + "_meta.json")
            with meta_path.open() as f:
                meta = json.load(f)
            for i, (seq, t_ns) in enumerate(zip(meta["seqs"], meta["t_ns"])):
                self._frames.append((seq, t_ns, arr[i]))
        logger.info(f"msg=replay_loaded frame_count={len(self._frames)} path={self._dir}")

    def read(self) -> ThermalFrame | None:
        if self._idx >= len(self._frames):
            return None
        seq, t_ns, img = self._frames[self._idx]
        self._idx += 1
        return ThermalFrame(seq=seq, t_capture_ns=t_ns, image=img.copy())

    @property
    def total_frames(self) -> int:
        return len(self._frames)


class ReplayImuSource:
    """Replays recorded IMU samples in order."""

    def __init__(self, session_dir: Path) -> None:
        self._dir = Path(session_dir)
        self._samples: list[ImuSample] = []
        self._idx = 0
        self._load()

    def _load(self) -> None:
        imu_path = self._dir / "imu.csv"
        with imu_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = (float(row["w"]), float(row["x"]), float(row["y"]), float(row["z"]))
                self._samples.append(ImuSample(t_ns=int(row["t_ns"]), quat_wxyz=q))
        logger.info(f"msg=imu_replay_loaded sample_count={len(self._samples)}")

    def read(self) -> ImuSample | None:
        if self._idx >= len(self._samples):
            return None
        s = self._samples[self._idx]
        self._idx += 1
        return s

    @property
    def total_samples(self) -> int:
        return len(self._samples)
