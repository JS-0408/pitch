# T6.2 — Unit tests for Orchestrator
import numpy as np
import pytest
from yaazhi.pipeline.orchestrator import Orchestrator
from yaazhi.types import SystemState


def test_orchestrator_initialization():
    orch = Orchestrator()
    assert orch is not None
    assert orch.safety_monitor.state == SystemState.NORMAL


def test_orchestrator_single_perception_step():
    orch = Orchestrator()
    orch.process_perception_step(timestamp_ms=100.0)
    assert orch.latest_raw_frame is not None
    assert orch.latest_proc_frame is not None
    assert orch.latest_proc_frame.shape == orch.latest_raw_frame.shape


def test_orchestrator_render_hud_frame():
    orch = Orchestrator()
    orch.process_perception_step(timestamp_ms=100.0)
    canvas = orch.render_hud_frame(timestamp_ms=116.6)
    assert isinstance(canvas, np.ndarray)
    assert canvas.ndim == 3
    assert canvas.shape[2] == 3  # BGR canvas


def test_orchestrator_short_benchmark():
    orch = Orchestrator()
    stats = orch.run_benchmark(duration_s=1.0)
    assert stats["perception_frames"] > 0
    assert stats["render_frames"] > 0
    assert stats["render_fps"] >= 30.0
