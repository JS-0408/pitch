"""
T1.2 — Sensor emulator and MockThermalSource.
Reads an image folder or video, downscales to the configured sensor resolution,
and paces output at the configured fps using a monotonic-clock pacer.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from yaazhi.types import ThermalFrame

logger = logging.getLogger(__name__)


def _load_frames_from_folder(folder: Path, sensor_h: int, sensor_w: int) -> list[np.ndarray]:
    """Load all images from a folder, resize to sensor dims."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"No image files found in {folder}")
    frames = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"msg=failed_to_read path={p}")
            continue
        resized = cv2.resize(img, (sensor_w, sensor_h), interpolation=cv2.INTER_AREA)
        frames.append(resized)
    logger.info(f"msg=loaded_frames count={len(frames)} path={folder}")
    return frames


def _load_frames_from_video(video: Path, sensor_h: int, sensor_w: int) -> list[np.ndarray]:
    """Extract all frames from a video file, convert to grayscale, resize."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        resized = cv2.resize(gray, (sensor_w, sensor_h), interpolation=cv2.INTER_AREA)
        frames.append(resized)
    cap.release()
    logger.info(f"msg=loaded_video_frames count={len(frames)} path={video}")
    return frames


def _quantize_14bit(frame: np.ndarray) -> np.ndarray:
    """Scale uint8 frame to simulate a 14-bit thermal sensor range (returned as uint8)."""
    f16 = frame.astype(np.float32) / 255.0
    raw14 = (f16 * 16383).astype(np.uint16)
    # Normalize back for display
    display = (raw14 >> 6).astype(np.uint8)
    return display


class MockThermalSource:
    """
    Emulates a Lepton-class (160x120, ~9 Hz) thermal camera.
    Paces frame emission using a monotonic-clock pacer with no sleep drift.
    Loops the source frames indefinitely.

    Implements FrameSource protocol:
        def read(self) -> ThermalFrame | None
    """

    def __init__(
        self,
        source: Path,
        sensor_h: int = 120,
        sensor_w: int = 160,
        fps: float = 9.0,
        quantize: bool = True,
        loop: bool = True,
    ) -> None:
        self._fps = fps
        self._period_ns = int(1e9 / fps)
        self._loop = loop
        self._seq = 0
        self._frame_idx = 0

        source = Path(source)
        if source.is_dir():
            self._frames = _load_frames_from_folder(source, sensor_h, sensor_w)
        elif source.is_file():
            self._frames = _load_frames_from_video(source, sensor_h, sensor_w)
        else:
            raise FileNotFoundError(f"Source not found: {source}")

        if quantize:
            self._frames = [_quantize_14bit(f) for f in self._frames]

        self._next_emit_ns: int = time.monotonic_ns()

    def read(self) -> ThermalFrame | None:
        """
        Block until the next frame is due, then return it.
        Returns None if source is exhausted and loop=False.
        """
        now = time.monotonic_ns()
        wait_ns = self._next_emit_ns - now
        if wait_ns > 0:
            time.sleep(wait_ns / 1e9)

        if self._frame_idx >= len(self._frames):
            if not self._loop:
                return None
            self._frame_idx = 0

        img = self._frames[self._frame_idx].copy()
        t_ns = time.monotonic_ns()
        frame = ThermalFrame(seq=self._seq, t_capture_ns=t_ns, image=img, raw=None)

        self._seq += 1
        self._frame_idx += 1
        self._next_emit_ns += self._period_ns

        return frame

    def __iter__(self) -> Iterator[ThermalFrame]:
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame


class SyntheticThermalSource:
    """
    Emits synthetic grayscale frames (gradient pattern) for testing
    without any real dataset.  Same pacer as MockThermalSource.
    """

    def __init__(
        self,
        sensor_h: int = 120,
        sensor_w: int = 160,
        fps: float = 9.0,
        num_frames: int = 0,  # 0 = infinite
    ) -> None:
        self._h = sensor_h
        self._w = sensor_w
        self._fps = fps
        self._period_ns = int(1e9 / fps)
        self._seq = 0
        self._max = num_frames
        self._next_emit_ns = time.monotonic_ns()

    def read(self) -> ThermalFrame | None:
        if self._max > 0 and self._seq >= self._max:
            return None

        now = time.monotonic_ns()
        wait_ns = self._next_emit_ns - now
        if wait_ns > 0:
            time.sleep(wait_ns / 1e9)

        # Animated gradient so frames are not identical
        t = self._seq / max(self._fps, 1)
        xs = np.linspace(0, 255, self._w, dtype=np.float32)
        ys = np.linspace(0, 255, self._h, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        img = ((xx + yy + t * 30) % 256).astype(np.uint8)

        t_ns = time.monotonic_ns()
        frame = ThermalFrame(seq=self._seq, t_capture_ns=t_ns, image=img)
        self._seq += 1
        self._next_emit_ns += self._period_ns
        return frame

    def __iter__(self) -> "Iterator[ThermalFrame]":
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame
