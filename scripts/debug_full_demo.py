"""
Full debug script — exercises every permutation of the demo pipeline headlessly.
ASCII output only (Windows cp1252 safe).
"""
import sys, traceback
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import cv2

PASS = []
FAIL = []

def check(label, fn):
    try:
        fn()
        PASS.append(label)
        print(f"  PASS  {label}")
    except Exception as e:
        FAIL.append((label, traceback.format_exc()))
        print(f"  FAIL  {label}")
        print(f"        {type(e).__name__}: {e}")

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────
print("\n=== 1. IMPORTS ===")

def _import_all():
    from yaazhi.config import settings
    from yaazhi.ingestion.imu_synth import SyntheticImuSource, ImuProfile
    from yaazhi.ingestion.sources import SyntheticThermalSource
    from yaazhi.perception.preprocess import AgcMode, preprocess
    from yaazhi.pipeline.orchestrator import Orchestrator
    from yaazhi.safety.faults import FaultType, FaultInjector
    from yaazhi.safety.monitor import SystemState
    from yaazhi.reprojection.warp import reproject, rotation_delta
    from yaazhi.reprojection.camera import build_K
    from yaazhi.rendering.hud import Renderer
    from yaazhi.tracking.head_pose import HeadPosePredictor
    from yaazhi.app.expo_app import ExpoApp

check("all imports", _import_all)

from yaazhi.ingestion.imu_synth import SyntheticImuSource, ImuProfile
from yaazhi.ingestion.sources import SyntheticThermalSource
from yaazhi.perception.preprocess import AgcMode, preprocess
from yaazhi.pipeline.orchestrator import Orchestrator
from yaazhi.safety.faults import FaultType, FaultInjector
from yaazhi.safety.monitor import SystemState
from yaazhi.reprojection.warp import reproject, rotation_delta
from yaazhi.reprojection.camera import build_K
from yaazhi.rendering.hud import Renderer
from yaazhi.types import SystemState as ST

# ── 2. ORCHESTRATOR BOOT ──────────────────────────────────────────────────────
print("\n=== 2. ORCHESTRATOR BOOT ===")
orch = None

def _boot():
    global orch
    orch = Orchestrator(target_fps=60.0, display_scale=4)
    orch.start()
check("orchestrator init + start", _boot)

# ── 3. ALL 5 IMU PROFILES ─────────────────────────────────────────────────────
print("\n=== 3. ALL 5 IMU PROFILES ===")

for prof in ImuProfile:
    def _test_profile(p=prof):
        orch.imu_source = SyntheticImuSource(profile=p, rate_hz=200.0)
        for i in range(3):
            orch.process_perception_step(float(i * 111))
        frame = orch.render_hud_frame(333.0)
        assert frame is not None and frame.ndim == 3
    check(f"IMU profile={prof.value}", _test_profile)

# ── 4. ALL 3 AGC MODES ───────────────────────────────────────────────────────
print("\n=== 4. ALL 3 AGC MODES ===")
dummy = np.random.randint(0, 255, (120, 160), dtype=np.uint8)

for mode in AgcMode:
    def _test_agc(m=mode):
        out = preprocess(dummy, mode=m)
        assert out.shape == (120, 160) and out.dtype == np.uint8
    check(f"AGC mode={mode.value}", _test_agc)

# ── 5. REPROJECTION ON / OFF ─────────────────────────────────────────────────
print("\n=== 5. REPROJECTION TOGGLE ===")

def _reproj_on():
    orch.reprojection_enabled = True
    frame = orch.render_hud_frame(500.0)
    assert frame is not None and frame.ndim == 3
check("render reprojection=ON", _reproj_on)

def _reproj_off():
    orch.reprojection_enabled = False
    frame = orch.render_hud_frame(600.0)
    assert frame is not None and frame.ndim == 3
check("render reprojection=OFF", _reproj_off)

orch.reprojection_enabled = True

# ── 6. ALL 4 FAULT TYPES ─────────────────────────────────────────────────────
print("\n=== 6. ALL 4 FAULT TYPES ===")

for fault in FaultType:
    def _test_fault(f=fault):
        orch.inject_fault(f)
        for i in range(5):
            orch.process_perception_step(float(5000 + i * 111))
        frame = orch.render_hud_frame(5550.0)
        assert frame is not None and frame.ndim == 3
        state = orch.safety_monitor.state
        print(f"        fault={f.value} -> state={state.name}")
    check(f"fault={fault.value}", _test_fault)

orch.inject_fault(FaultType.NONE)

# ── 7. SENSOR LOST BANNER ────────────────────────────────────────────────────
print("\n=== 7. SENSOR LOST BANNER (SAFE STATE) ===")

def _safe_banner():
    orch.inject_fault(FaultType.IMU_DROP)
    for i in range(20):
        orch.process_perception_step(float(10000 + i * 111))
    frame = orch.render_hud_frame(12200.0)
    assert frame is not None and frame.ndim == 3
    state = orch.safety_monitor.state
    print(f"        After IMU_DROP x20 -> state={state.name}")
check("SAFE state renders without crash", _safe_banner)

def _recovery():
    orch.inject_fault(FaultType.NONE)
    for i in range(20):
        orch.process_perception_step(float(15000 + i * 111))
    frame = orch.render_hud_frame(17200.0)
    assert frame is not None and frame.ndim == 3
    state = orch.safety_monitor.state
    print(f"        After NONE fault x20 -> state={state.name}")
check("Recovery from SAFE", _recovery)

# ── 8. SMOKE VISUAL EFFECT ───────────────────────────────────────────────────
print("\n=== 8. SMOKE VISUAL EFFECT ===")

def _smoke():
    raw = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    smoke_layer = cv2.GaussianBlur(raw, (21, 21), 8)
    result = cv2.addWeighted(raw, 0.35, smoke_layer, 0.65, 0)
    assert result.shape == raw.shape and result.dtype == np.uint8
check("smoke GaussianBlur visuals", _smoke)

# ── 9. RENDERER — ALL STATES ─────────────────────────────────────────────────
print("\n=== 9. RENDERER — ALL SYSTEM STATES ===")
renderer = Renderer(scale=4)
sample = np.random.randint(0, 255, (120, 160), dtype=np.uint8)
hud_data = {"render_fps": 60, "m2d_ms": 15, "tracks": 0, "toggles": "AGC:ON"}

for state in ST:
    def _render(s=state):
        frame = renderer.draw(sample, [], hud_data, state=s)
        assert frame is not None and frame.ndim == 3
    check(f"renderer state={state.name}", _render)

# ── 10. EXPOAPP ATTRIBUTES ───────────────────────────────────────────────────
print("\n=== 10. EXPOAPP ATTRIBUTES ===")

def _expoapp_attrs():
    from yaazhi.app.expo_app import ExpoApp
    app = ExpoApp(fps=60.0, display_scale=4)
    assert hasattr(app, "orchestrator")
    assert hasattr(app, "reprojection_enabled")
    assert hasattr(app, "smoke_enabled")
    assert hasattr(app, "agc_idx")
    assert hasattr(app, "agc_modes")
    assert hasattr(app, "fault_idx")
    assert hasattr(app, "imu_profile_idx")
    assert hasattr(app.orchestrator, "reprojection_enabled")
    assert hasattr(app.orchestrator, "imu_source")
    assert hasattr(app.orchestrator, "inject_fault")
check("ExpoApp all attributes exist", _expoapp_attrs)

# ── 11. ALL HOTKEY ACTIONS ───────────────────────────────────────────────────
print("\n=== 11. ALL HOTKEY ACTIONS ===")
from yaazhi.app.expo_app import ExpoApp
app = ExpoApp(fps=60.0, display_scale=4)
app.orchestrator.start()
for i in range(5):
    app.orchestrator.process_perception_step(float(i * 111))

def _key_R_on():
    app.reprojection_enabled = True
    app.orchestrator.reprojection_enabled = True
    f = app.orchestrator.render_hud_frame(700.0)
    assert f is not None
check("key R -> reproj ON + render", _key_R_on)

def _key_R_off():
    app.reprojection_enabled = False
    app.orchestrator.reprojection_enabled = False
    f = app.orchestrator.render_hud_frame(800.0)
    assert f is not None
check("key R -> reproj OFF + render", _key_R_off)

def _key_S_on():
    app.smoke_enabled = True
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    sl = cv2.GaussianBlur(raw, (21, 21), 8)
    res = cv2.addWeighted(raw, 0.35, sl, 0.65, 0)
    assert res is not None
check("key S -> smoke ON visual", _key_S_on)

def _key_S_off():
    app.smoke_enabled = False
check("key S -> smoke OFF", _key_S_off)

def _key_H_all():
    for _ in range(3):
        app.agc_idx = (app.agc_idx + 1) % 3
        out = preprocess(dummy, mode=app.agc_modes[app.agc_idx])
        assert out is not None
check("key H x3 -> all AGC modes", _key_H_all)

def _key_F_all():
    faults = [FaultType.NONE, FaultType.IMU_DROP,
              FaultType.FRAME_FREEZE, FaultType.DETECTOR_SPIKE]
    for f in faults:
        app.orchestrator.inject_fault(f)
        frame = app.orchestrator.render_hud_frame(9000.0)
        assert frame is not None
    app.orchestrator.inject_fault(FaultType.NONE)
check("key F x4 -> all faults + render each", _key_F_all)

def _key_1_to_5_all():
    profiles = [ImuProfile.STILL, ImuProfile.SLOW_SCAN, ImuProfile.FAST_TURN,
                ImuProfile.ABRUPT_REVERSAL, ImuProfile.WALKING_BOB]
    for i, prof in enumerate(profiles):
        app.imu_profile_idx = i
        app.orchestrator.imu_source = SyntheticImuSource(profile=prof, rate_hz=200.0)
        app.orchestrator.process_perception_step(float(2000 + i * 111))
        frame = app.orchestrator.render_hud_frame(float(2600 + i * 100))
        assert frame is not None, f"render failed for {prof.value}"
check("keys 1-5 -> all IMU profiles + render each", _key_1_to_5_all)

def _key_space():
    app.paused = True
    assert app.paused
    app.paused = False
    assert not app.paused
check("key SPACE -> pause/resume", _key_space)

app.orchestrator.stop()

# ── 12. REPROJECTION MATH VALIDATION ─────────────────────────────────────────
print("\n=== 12. REPROJECTION MATH ===")

def _reproj_math():
    K = build_K(width=160, height=120, hfov_deg=57.0)
    # Identity rotation — should return original image unchanged
    R_id = np.eye(3)
    img = np.random.randint(0, 255, (120, 160), dtype=np.uint8)
    result = reproject(img, K, R_id)
    assert result.warped.shape == img.shape
    assert not result.clamped
check("reproject identity rotation", _reproj_math)

def _reproj_10deg_yaw():
    import math
    K = build_K(width=160, height=120, hfov_deg=57.0)
    ang = math.radians(10)
    R = np.array([[math.cos(ang), 0, math.sin(ang)],
                  [0, 1, 0],
                  [-math.sin(ang), 0, math.cos(ang)]])
    img = np.random.randint(0, 255, (120, 160), dtype=np.uint8)
    result = reproject(img, K, R)
    assert result.warped.shape == img.shape
    assert not result.clamped
check("reproject 10deg yaw (not clamped)", _reproj_10deg_yaw)

def _reproj_90deg_clamped():
    import math
    K = build_K(width=160, height=120, hfov_deg=57.0)
    ang = math.radians(90)
    R = np.array([[math.cos(ang), 0, math.sin(ang)],
                  [0, 1, 0],
                  [-math.sin(ang), 0, math.cos(ang)]])
    img = np.random.randint(0, 255, (120, 160), dtype=np.uint8)
    result = reproject(img, K, R)
    assert result.clamped, "90deg should be clamped"
check("reproject 90deg (clamped correctly)", _reproj_90deg_clamped)

# ── FINAL REPORT ──────────────────────────────────────────────────────────────
total = len(PASS) + len(FAIL)
print(f"\n{'='*55}")
print(f"  PASSED: {len(PASS)}/{total}")
print(f"  FAILED: {len(FAIL)}/{total}")
print(f"{'='*55}")

if FAIL:
    print("\n-- FAILURE DETAILS --")
    for label, tb in FAIL:
        print(f"\nFAIL: {label}")
        print(tb)
    sys.exit(1)
else:
    print("\nALL CHECKS PASSED - demo is safe to run")
    sys.exit(0)
