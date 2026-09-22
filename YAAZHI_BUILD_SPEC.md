# YAAZHI: Agent Build Spec and Tracker

**Purpose:** a precise, task-by-task specification an AI coding agent (Antigravity) can execute, and a tracker both you and the agent update. Goal: a **fully working software prototype, demoable live at the expo**, built without physical hardware.

**How to use:** place this file at the repo root as `AGENTS.md` (or paste Section 3 into your agent's rules/instructions; I haven't verified which file Antigravity reads automatically). Then copy `docs/TRACKER.md` (Section 11) into the repo. The agent works one task ID at a time, in order, and updates the tracker with evidence.

**Kickoff prompt for the agent:**
> Read AGENTS.md fully. Work only on the next unchecked task in docs/TRACKER.md. Implement it to the spec, run its acceptance commands, paste the real output into the tracker, commit, then stop and report. Never invent metrics. If blocked, say exactly what you need.

---

## 1. Mission, deliverable, non-goals

**One-line product:** hands-free thermal-vision assistance software: detects people in low-resolution thermal video, keeps markers stable under head motion using IMU reprojection, and degrades safely when inputs fail.

**Expo deliverable (what must run on your laptop, offline):**
1. A single window showing **raw thermal (choppy, low-rate)** next to **Yaazhi output (smooth, reprojected, markers)** with live metrics.
2. Live toggles: reprojection on/off, smoke simulation on/off, hot-scene handling on/off, sensor-fault injection.
3. A recorded-session replay viewer (after-action review).
4. A one-page evaluation report with **measured** numbers.
5. A scripted 3-minute demo scenario plus a pre-rendered fallback video.

**Non-goals (do not build):** hardware drivers, Jetson/TensorRT, optical combiner code, video over radio, flashover/collapse prediction, military features, fire-rating claims.

**Honesty rules for the demo:** footage is public thermal data (labelled as such), IMU is synthetic (labelled), smoke is simulated (labelled). Nothing on screen may imply fire-rated or field-validated performance.

---

## 2. Environment and resource decisions

| Resource | Spec | Decision |
|---|---|---|
| Local | Ryzen 5 5600, GTX 1650 4 GB, 8 GB RAM | Primary dev + expo machine |
| Cloud | Azure student credit, about $100 | Optional; see Section 10 |
| OS | Windows | **Develop and demo natively on Windows** (Python venv, CUDA PyTorch). Skip WSL2: with 8 GB RAM its overhead hurts, and TensorRT/GStreamer are no longer needed |
| Inference | ONNX Runtime (CUDA provider, CPU fallback) | **No TensorRT for the prototype.** It is a hardware-track task |
| Model | YOLOv8n (Ultralytics), input 320 | Trained on emulated low-res thermal |
| Python | 3.10 or 3.11 | Pin all versions in `requirements.txt` |
| UI | OpenCV window (HighGUI) | Zero extra deps, reliable on any expo laptop |

**Memory rules (8 GB RAM):** dataloader `workers=2`, batch size adjusted to fit 4 GB VRAM, close browsers during training, keep frames as uint8/uint16 numpy, no full-dataset-in-RAM caching.

**Sensor emulation (key design choice):** the real target sensor is Lepton-class (160x120, about 57 degree horizontal FOV, about 9 Hz; verify against datasheet). All datasets are **downscaled to 160x120 and frame-rate-limited to 9 Hz** so results are relevant to the target sensor. Person size in pixels: `f_px = 80 / tan(28.5 deg) = 147`; person 1.7 m tall at distance d gives height about `147 * 1.7 / d` px (10 m: about 25 px, 20 m: about 12 px, 30 m: about 8 px). Distance is reported via this pixel-height mapping.

---

## 3. Ground rules for the agent (Definition of Done)

1. One task at a time, in tracker order. Do not start a task until its dependencies are checked.
2. A task is done only when: code committed, unit tests pass (`pytest -q`), acceptance commands run, **real output pasted** in the tracker's evidence column.
3. **Never fabricate or estimate a metric.** If it wasn't measured by a script, it does not go in the tracker, docs, or UI.
4. No hardcoded paths, resolutions, thresholds, or rates in module code. Everything in `configs/*.yaml`.
5. All timestamps use `time.monotonic_ns()` on one clock. Every frame and IMU sample carries one.
6. Use the `logging` module with structured key=value messages. No `print` outside CLI entry points.
7. Type hints on all public functions. Public interfaces match Section 4 exactly. If a change is needed, edit the spec and log it in `docs/decisions.md` with a one-line reason.
8. Every module has a pure-function core that is unit-testable without a GPU or window.
9. Failures must degrade safely: bad input is rejected and logged, never passed downstream; lost input fades markers, never freezes them at full confidence.
10. Commit message format: `T<id>: <what>`. One commit per task minimum.
11. Do not add dependencies without adding them to `requirements.txt` and noting why in `docs/decisions.md`.

---

## 4. Architecture and data contracts

### 4.1 Pipeline

```
MockThermalSource (9 Hz) --\
                            >--> Ingestion/Validation --> latest-frame slot
MockImuSource (200 Hz)   --/                              |
                                                          v
                               Inference thread (only on NEW frames, about 9 Hz)
                               preprocess -> detector -> tracker.update(detections)
                                                          |
                                                          v
              Render loop (60 Hz, never blocks) --------- reads:
              1. latest IMU pose, predicted to display time
              2. latest frame -> reprojection warp
              3. tracks predicted forward to display time
              4. overlay + metrics -> window / recorder
                                                          |
                         Safety monitor watches all inputs, can force SAFE state
```

**The render loop is the fast loop.** It runs at a fixed rate regardless of inference speed. Inference is the slow loop, rate-limited by the sensor.

### 4.2 Data types (`src/yaazhi/types.py`)

```python
from dataclasses import dataclass
from enum import Enum
import numpy as np

@dataclass(frozen=True)
class ThermalFrame:
    seq: int
    t_capture_ns: int
    image: np.ndarray          # HxW uint8 (display) ; raw uint16 kept in .raw if present
    raw: np.ndarray | None = None

@dataclass(frozen=True)
class ImuSample:
    t_ns: int
    quat_wxyz: tuple[float, float, float, float]   # head orientation, world frame

@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    conf: float
    cls: int                    # 0 = person

class TrackStatus(Enum):
    TENTATIVE = 1
    CONFIRMED = 2
    COASTING = 3               # measurement missing, predicting
    LOST = 4

@dataclass
class Track:
    id: int
    status: TrackStatus
    cx: float; cy: float; w: float; h: float
    vx: float; vy: float
    conf: float                 # 0..1, decays while COASTING
    t_last_update_ns: int

@dataclass(frozen=True)
class Marker:
    track_id: int
    x: float; y: float; w: float; h: float
    alpha: float                # 0..1 from confidence
    label: str

class SystemState(Enum):
    NORMAL = 1
    DEGRADED = 2                # e.g., stale thermal or IMU gaps: markers faded
    SAFE = 3                    # sensors lost: overlay off, "CLEAR VIEW" banner
```

### 4.3 Module interfaces (must match)

```python
# ingestion
class FrameSource(Protocol):
    def read(self) -> ThermalFrame | None: ...      # blocking with timeout
class ImuSource(Protocol):
    def read(self) -> ImuSample | None: ...

# perception
class Detector(Protocol):
    def detect(self, frame: ThermalFrame) -> list[Detection]: ...

# reprojection
def rotation_delta(q_from, q_to) -> np.ndarray: ...            # 3x3
def reproject(image: np.ndarray, K: np.ndarray, R_delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """returns (warped_image, valid_mask)"""

# tracking
class HeadPosePredictor:
    def update(self, s: ImuSample) -> None: ...
    def predict(self, t_ns: int) -> np.ndarray: ...            # quaternion
class TargetTracker:
    def update(self, dets: list[Detection], t_ns: int) -> list[Track]: ...
    def predict(self, t_ns: int) -> list[Track]: ...

# rendering
class Renderer:
    def draw(self, image: np.ndarray, markers: list[Marker], hud: dict) -> np.ndarray: ...

# safety
class SafetyMonitor:
    def observe(self, *, t_frame_ns, t_imu_ns, now_ns, fps, ...) -> SystemState: ...
```

---

## 5. Repository layout

```
yaazhi-vision/
├── AGENTS.md                      # this file
├── README.md                      # how to run the demo in one command
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── local.yaml                 # default; expo laptop
│   └── azure.yaml                 # training-only overrides
├── src/yaazhi/
│   ├── types.py
│   ├── config.py                  # loads yaml, single env flag
│   ├── logging_setup.py
│   ├── ingestion/  (sources.py, validation.py, sensor_emulator.py, imu_synth.py, recorder.py)
│   ├── perception/ (dataset_prep.py, detector_onnx.py, thresholds.py, preprocess.py)
│   ├── reprojection/ (camera.py, warp.py)
│   ├── tracking/   (head_pose.py, target_tracker.py, association.py)
│   ├── rendering/  (hud.py, palette.py)
│   ├── safety/     (monitor.py, faults.py)
│   ├── comms/      (alerts.py, command_post.py)
│   ├── pipeline/   (orchestrator.py, latency.py)
│   └── app/        (expo_app.py, replay_viewer.py, scenario.py)
├── scripts/
│   ├── prepare_data.py  train.py  export_onnx.py  evaluate.py
│   ├── run_demo.py  run_replay.py  azure_vm.ps1  make_fallback_video.py
├── tests/  (unit/, integration/)
├── data/   (gitignored: raw/, processed/, sessions/)
├── models/ (gitignored large files; keep model card .md)
├── reports/ (eval_*.json, eval_report.md)
└── docs/   (TRACKER.md, decisions.md, perf_history.md)
```

---

## 6. Config (`configs/local.yaml`, required keys)

```yaml
sensor:            {width: 160, height: 120, hfov_deg: 57.0, fps: 9}
imu:               {rate_hz: 200}
display:           {render_fps: 60, window_scale: 4}
detector:          {model_path: models/yaazhi_n320.onnx, imgsz: 320, conf: 0.25, nms_iou: 0.5, provider: cuda}
tracker:           {confirm_hits: 3, coast_ms: 600, drop_ms: 1500, gate_iou: 0.2, conf_decay_per_s: 1.2}
reprojection:      {enabled: true, max_delta_deg: 60}
safety:            {frame_stale_ms: 400, imu_stale_ms: 100, fade_after_ms: 300, safe_after_ms: 1200}
preprocess:        {agc: adaptive, clahe_clip: 2.0}
smoke_sim:         {enabled: false, density: 0.6}
paths:             {data: data, models: models, reports: reports, sessions: data/sessions}
```

---

## 7. Task list

Each task: **Goal / Files / Spec / Acceptance / Evidence**. Dependencies are listed as `Needs`.

### Phase 0: Setup

**T0.1 Repo, environment, tooling.** Needs: none.
- Files: repo skeleton (Section 5), `requirements.txt`, `pyproject.toml`, `.gitignore` (data/, models/*.pt, *.onnx, .env, sessions/).
- Spec: Python venv, `pytest`, `ruff`. CUDA PyTorch installed and `torch.cuda.is_available()` true.
- Acceptance: `python -c "import torch;print(torch.cuda.is_available())"` prints True; `pytest -q` runs (0 tests OK).

**T0.2 Config and logging.** Needs: T0.1.
- Files: `config.py`, `logging_setup.py`, `configs/local.yaml`.
- Spec: typed config object loaded from YAML; env flag `YAAZHI_ENV=local|azure`; missing key raises clear error. Logger emits `key=value`.
- Acceptance: unit tests for load, override, missing-key error.

**T0.3 Decisions log.** Needs: T0.1. Create `docs/decisions.md` seeded with: Windows-native, ONNX-not-TensorRT, sensor emulation at 160x120/9 Hz, render-loop-as-fast-loop.

### Phase 1: Contracts and mock ingestion

**T1.1 Types.** Needs: T0.2. Implement Section 4.2 exactly. Acceptance: unit test instantiates all types, frozen classes reject mutation.

**T1.2 Sensor emulator + dataset frame source.** Needs: T1.1.
- Files: `ingestion/sensor_emulator.py`, `ingestion/sources.py` (`MockThermalSource`).
- Spec: reads an image folder or video; converts to grayscale, resizes to 160x120 (area interpolation), optionally quantizes to 14-bit-like range; emits at configured fps using a monotonic-clock pacer (no `sleep` drift accumulation); stamps `t_capture_ns`.
- Acceptance: run for 30 s; measured rate within +/-2% of 9 Hz; max inter-frame jitter logged. Test uses a temp folder of synthetic frames.

**T1.3 Synthetic IMU generator.** Needs: T1.1.
- Files: `ingestion/imu_synth.py`.
- Spec: generates head orientation quaternions at 200 Hz from scripted profiles: `still`, `slow_scan` (30 deg/s), `fast_turn` (200 deg/s smooth), `abrupt_reversal`, `walking_bob`. Adds configurable gyro noise. Deterministic with a seed. Also exposes ground truth orientation at any time `t`.
- Acceptance: unit tests check peak angular rate per profile and unit-norm quaternions.

**T1.4 Validation and reconnect.** Needs: T1.2, T1.3.
- Files: `ingestion/validation.py`.
- Spec: reject frames with wrong shape/dtype/NaN or non-monotonic timestamps; reject quaternions with norm outside [0.99, 1.01] or angular rate above physical limit (config). Wrap sources in a retrying supervisor: on source exception, log, back off (0.1, 0.2, 0.5, 1 s), and resume. Count and log every rejection/drop.
- Acceptance: unit tests for each rejection type; a fault-injecting source that raises 3 times recovers without crashing.

**T1.5 Recorder.** Needs: T1.4.
- Files: `ingestion/recorder.py`.
- Spec: session format = directory with `frames.npy` chunks or PNGs, `imu.csv`, `markers.jsonl`, `meta.json` (config snapshot, git hash). Deterministic replay via `ReplayFrameSource`/`ReplayImuSource` implementing the same protocols.
- Acceptance: record 10 s, replay, assert frame count and timestamps identical.

### Phase 2: Data and detector

**T2.1 Dataset preparation.** Needs: T1.2. **Human prerequisite:** download datasets (Section 14).
- Files: `perception/dataset_prep.py`, `scripts/prepare_data.py`.
- Spec: convert to YOLO format with person class only; downscale each image and its boxes to 160x120 (emulated), and store a 320x240 upscaled copy (bicubic) as training input so the model sees the same blur it will see at runtime; drop boxes smaller than a configured minimum (log how many); fixed-seed train/val/test split **by sequence/scene, not random frames** (avoids leakage). Write `data/processed/split_manifest.json` and dataset statistics (box-height histogram).
- Acceptance: script prints counts per split and box-height histogram; a unit test checks box scaling math.

**T2.2 Baseline training (local).** Needs: T2.1.
- Files: `scripts/train.py`.
- Spec: Ultralytics YOLOv8n, `imgsz=320`, batch size auto-fit to 4 GB, `workers=2`, AMP on, fixed seed, augmentations: flip, brightness/contrast jitter, gain-saturation jitter, small blur. Log to `runs/`.
- Acceptance: training completes; `models/best.pt` exists; validation mAP printed. If local VRAM/time is insufficient, use Section 10.

**T2.3 Evaluation harness.** Needs: T2.2.
- Files: `scripts/evaluate.py`, `reports/eval_*.json`.
- Spec: on the **test split**, compute per person-height bucket (px): `<8, 8-12, 12-16, 16-25, >25` (mapped to approximate distances via `f_px`): recall, precision, false positives per frame. Output JSON plus a markdown table. Also report inference latency (median, p95) over 500 frames.
- Acceptance: `python scripts/evaluate.py --model models/best.pt` writes report. **This is the first real number for the pitch.**

**T2.4 ONNX export + runtime wrapper.** Needs: T2.2.
- Files: `scripts/export_onnx.py`, `perception/detector_onnx.py`.
- Spec: export with fixed input 320, FP16 where supported; `ONNXDetector` implements `Detector`, uses CUDA provider with CPU fallback, does its own letterbox, NMS with config thresholds. Parity test: ONNX boxes match PyTorch within tolerance on 50 images.
- Acceptance: parity test passes; latency measured and logged to `docs/perf_history.md`.

**T2.5 Threshold tuning, logged.** Needs: T2.3, T2.4.
- Spec: sweep `conf` and `nms_iou`, record precision/recall curve, choose operating point that meets a false-alarm budget defined in config (default: at most 1 false positive per minute at 9 Hz, i.e. FP per frame <= 0.002). Save curve to `reports/threshold_sweep.json` and record the choice in `docs/decisions.md`. Do **not** leave library defaults.
- Acceptance: chosen thresholds written to `configs/local.yaml` with the report path referenced.

### Phase 3: IMU reprojection

**T3.1 Camera model.** Needs: T1.1.
- Files: `reprojection/camera.py`.
- Spec: pinhole intrinsics `K` from width/height/hfov; helpers for pixel-to-ray and ray-to-pixel.
- Acceptance: unit tests (center pixel maps to optical axis; round-trip error < 1e-6).

**T3.2 Rotation-delta warp.** Needs: T3.1, T1.3.
- Files: `reprojection/warp.py`.
- Spec: `H = K @ R_delta @ K^-1`; warp with `cv2.warpPerspective`, return valid-region mask; clamp `|delta|` to `max_delta_deg` (beyond that, return original with a flag). Render black borders where invalid (do not smear).
- Acceptance: unit tests: zero delta is identity; a 10 degree yaw shifts the image center by about `f_px * tan(10 deg)` px (+/-1 px).

**T3.3 Reprojection quality evaluation.** Needs: T3.2, T1.2.
- Files: `scripts/eval_reprojection.py` (reuse `evaluate` reporting style).
- Spec: source video at full rate provides truth. For each IMU profile: stale frame captured at `t_c`; display time `t_d = t_c + delay` (delay sweep 0-150 ms). Truth view = frame nearest `t_d` with orientation `R(t_d)` applied. Compare **no reprojection** (stale frame at `R(t_c)`) vs **reprojection** (stale frame warped by `R(t_c)^-1 R(t_d)`), on the valid-mask region. Metrics: mean absolute error, and ground-truth box center error in px. Output table per profile.
- Acceptance: report written; reprojection error must be lower than no-reprojection for every non-`still` profile, or the failure is documented honestly in the report.

**T3.4 Latency accounting.** Needs: T3.2.
- Files: `pipeline/latency.py`.
- Spec: tag every displayed frame with `t_imu_used_ns`, `t_frame_capture_ns`, `t_display_ns`. Compute `motion_to_display = t_display - t_imu_used` (software-only; excludes display hardware; say so in the report). Rolling median/p95 + frame-age stat.
- Acceptance: unit test with injected timestamps.

### Phase 4: Tracking

**T4.1 Head-pose predictor.** Needs: T1.3.
- Files: `tracking/head_pose.py`.
- Spec: constant-velocity Kalman filter on yaw/pitch/roll (small-angle, wrap-safe) or on quaternion with angular velocity state; `predict(t_ns)` extrapolates to display time; extrapolation horizon capped (config). Independent class, no dependency on target tracker.
- Acceptance: tests: steady 100 deg/s turn predicted 50 ms ahead within 0.5 deg; abrupt reversal: report the overshoot number honestly in test output (document limitation).

**T4.2 Target tracker.** Needs: T1.1.
- Files: `tracking/target_tracker.py`, `tracking/association.py`.
- Spec: constant-velocity Kalman per track (state: cx, cy, vx, vy; w, h smoothed); association by IoU with Hungarian assignment plus gating; track lifecycle: `TENTATIVE` (needs `confirm_hits`) -> `CONFIRMED` -> `COASTING` on missed detection (predict only, confidence decays at `conf_decay_per_s`) -> `LOST` after `drop_ms`. `predict(t_ns)` extrapolates for the render loop without mutating filter state.
- Acceptance: synthetic tests: (a) constant-velocity target recovered with RMSE under a set bound; (b) target disappears -> COASTING -> LOST at the configured times; (c) two crossing targets keep IDs; (d) single false-positive frame never becomes CONFIRMED.

**T4.3 Tracker on real detections.** Needs: T4.2, T2.4.
- Spec: run detector + tracker over test sequences; report ID switches and track fragmentation counts (simple counts, not full MOT metrics).
- Acceptance: numbers written to `reports/tracking_eval.json`.

### Phase 5: Preprocessing and smoke simulation

**T5.1 Hot-scene handling.** Needs: T1.2.
- Files: `perception/preprocess.py`.
- Spec: three modes: `naive` (min-max stretch), `percentile` (clip 1-99.5%), `adaptive` (percentile + CLAHE with hot-pixel exclusion so a saturated blob does not compress the rest). Provide `inject_hot_blob(image, size, temp)` to synthesize saturation for tests.
- Acceptance: on frames with an injected hot blob, report local contrast (RMS contrast in person boxes) for the three modes; `adaptive` must be reported alongside `naive` in `reports/agc_eval.md`, whatever the result.

**T5.2 Smoke simulator.** Needs: T1.2.
- Files: `ingestion/sensor_emulator.py` (extend) or `perception/smoke_sim.py`.
- Spec: for a paired **visible** channel (if the dataset provides it, e.g. LLVIP), apply density-controlled haze (low-frequency noise + contrast reduction + blur) to show visible degradation. Thermal gets only mild attenuation and noise. Label clearly in UI: `SIMULATED SMOKE`.
- Acceptance: unit test that density 0 is identity; visual sample saved to `reports/smoke_sim_examples.png`.

### Phase 6: Rendering, orchestration, safety

**T6.1 HUD renderer.** Needs: T1.1.
- Files: `rendering/hud.py`, `rendering/palette.py`.
- Spec: ironbow-style palette for grayscale thermal; corner-bracket reticles (not full outlines); marker alpha from track confidence; COASTING markers drawn dashed; status bar (fps, latency median/p95, tracks, system state, mode toggles). Renderer has no knowledge of the output target. Output is a numpy image.
- Acceptance: unit test renders markers to an array and checks pixel presence at expected locations; snapshot images saved to `reports/`.

**T6.2 Orchestrator.** Needs: T3.2, T4.1, T4.2, T6.1, T2.4.
- Files: `pipeline/orchestrator.py`.
- Spec: threads: source reader, inference (processes only the newest frame, drops stale ones, logs drops), render loop at fixed `render_fps`. Shared state via lock-protected latest-value slots (not unbounded queues). The render loop never waits on inference. Graceful shutdown.
- Acceptance: integration test with mock sources runs 60 s: render fps within 5% of target, zero unhandled exceptions, latency report emitted.

**T6.3 Safety monitor and fault injection.** Needs: T6.2.
- Files: `safety/monitor.py`, `safety/faults.py`.
- Spec: state machine `NORMAL -> DEGRADED -> SAFE` based on `frame_stale_ms`, `imu_stale_ms`, inference failure, fps collapse. DEGRADED fades markers; SAFE clears the overlay and shows `SENSOR LOST. DIRECT VIEW.` Recovery requires N consecutive good samples (hysteresis). Fault injectors: drop IMU, freeze thermal, spike detector latency, corrupt frame.
- Acceptance: tests for each transition and for hysteresis; integration test injects each fault and asserts final state and that no stale marker stays at full alpha.

**T6.4 Alert messaging and command post.** Needs: T6.2.
- Files: `comms/alerts.py`, `comms/command_post.py`.
- Spec: compact binary alert (<= 32 bytes: id, timestamp, state, track count, sequence) sent over localhost UDP with a simulated lossy link (configurable drop %). A separate small console/window "command post" displays received status and highlights missing heartbeats. Explicitly no video.
- Acceptance: round-trip encode/decode tests; with 30% simulated loss, command post still shows correct latest state.

### Phase 7: Expo application

**T7.1 Expo app.** Needs: T6.2, T6.3, T5.1, T5.2.
- Files: `app/expo_app.py`, `scripts/run_demo.py`.
- Spec: one OpenCV window, three panels: left = raw emulated thermal at 9 Hz (choppy), right = Yaazhi output, bottom = live metrics. Hotkeys: `R` reprojection, `S` smoke sim, `H` hot-scene handling mode cycle, `F` fault injection cycle, `1-5` IMU profile, `SPACE` pause, `REC` (`C`) record, `Q` quit. On-screen legend lists hotkeys. Banner: `PUBLIC DATA + SYNTHETIC IMU`.
- Acceptance: `python scripts/run_demo.py` starts in under 10 s on the expo laptop and runs 10 minutes without crash or memory growth over 20%.

**T7.2 Replay viewer (after-action).** Needs: T1.5, T6.1.
- Files: `app/replay_viewer.py`.
- Spec: loads a session; timeline trackbar, play/pause, step frame, marker overlay toggle, track-ID labels, export clip as mp4.
- Acceptance: record in T7.1, open in replay viewer, scrub, export a clip.

**T7.3 Scripted scenario.** Needs: T7.1.
- Files: `app/scenario.py`.
- Spec: `--scenario expo` plays a fixed 3-minute sequence: (1) normal view; (2) fast head turn with reprojection off then on; (3) smoke sim on; (4) hot blob; (5) IMU dropout -> SAFE state -> recovery; (6) end with metrics summary. Deterministic (seeded).
- Acceptance: three consecutive runs produce identical event timelines (logged).

**T7.4 Fallback video.** Needs: T7.3.
- Files: `scripts/make_fallback_video.py`.
- Spec: renders the scripted scenario headless to `demo_fallback.mp4` with the same UI, so a laptop or projector failure never kills the demo.
- Acceptance: file created, plays in a standard player.

### Phase 8: Hardening and evidence

**T8.1 Performance history.** Needs: T6.2. Script appends a row to `docs/perf_history.md` per run: date, git hash, environment, render fps, inference median/p95, motion-to-display median/p95, VRAM peak, frame drops.

**T8.2 One-command run and README.** Needs: T7.4. `README.md` with setup (exact commands), how to run demo, replay, evaluation; troubleshooting for GPU missing (CPU fallback).

**T8.3 Evaluation report.** Needs: T2.5, T3.3, T4.3, T5.1. Compile `reports/eval_report.md`: detector table by size/distance, threshold choice, reprojection results, tracking counts, AGC comparison, latency, and a **Limitations** section written from the actual results.

**T8.4 Full dry runs.** Needs: all. Run the scripted demo three times on the expo laptop on battery and on power; fix anything that fails. Freeze with git tag `expo-v1`.

---

## 8. Metrics and targets

Targets are **hypotheses to test, not promises**. Replace with measured values in the tracker.

| Metric | Measured by | Initial target (hypothesis) |
|---|---|---|
| Render fps | T6.2 integration | 60 (stable, >= 55) |
| Inference latency (GTX 1650, 320 input) | T2.4 | < 15 ms median |
| Motion-to-display, software only | T3.4 | < 30 ms median |
| Reprojection benefit | T3.3 | lower error than no-reprojection on all moving profiles |
| Person recall by height bucket | T2.3 | report curve; no pre-promised number |
| False alarms | T2.5 | <= 1 per minute at chosen threshold |
| Track fragmentation / ID switches | T4.3 | report counts |
| Fault handling | T6.3 | all injected faults reach expected state |
| Demo stability | T7.1 | 10 min no crash |

---

## 9. Expo demo design

**Layout:** `[ raw thermal 9 Hz | Yaazhi output ]` above a metrics bar. Metrics: fps, latency median/p95, active tracks, system state, current toggles, and the label `PUBLIC DATA + SYNTHETIC IMU`.

**3-minute story:**
1. (0:00) "Thermal sees through smoke, but it's low-resolution and slow." Show raw vs. output.
2. (0:30) Fast head turn: reprojection off (judder, lag), then on (smooth). Show measured latency numbers.
3. (1:15) Smoke sim on: visible feed degrades, thermal detections persist. State clearly it is simulated.
4. (1:45) Hot blob: naive vs. adaptive contrast.
5. (2:15) Kill the IMU: markers fade, `SENSOR LOST. DIRECT VIEW.`, then recovery. "It fails safe."
6. (2:45) Open evaluation report: the detection range curve and limitations.

**Say out loud:** what is real (pipeline, measured numbers), what is simulated (IMU, smoke), what is not done (hardware, fire rating, certification).

**Backups:** `demo_fallback.mp4`, a second copy of the repo on a USB drive, `--cpu` flag, pre-generated report PDF/markdown.

---

## 10. Azure plan ($100 credit)

**Principle:** local first. At 320 input, YOLOv8n training on a low-resolution dataset may fit on the GTX 1650 (verify in T2.2). Use Azure only for what local can't do well.

| Use | When | Est. cost (verify current prices) |
|---|---|---|
| Longer training or multi-seed sweeps | If local run is too slow or OOM | T4 VM (NC4as_T4_v3), roughly $0.5/h on-demand, less on spot: budget <= 30 GPU-hours |
| Full-resolution comparison run | Optional, for a "why we downscale" slide | 5-10 GPU-hours |
| Benchmark on T4 | Optional | 1-2 hours |

- **Day 1:** confirm the credit is visible and check **GPU quota**. Student subscriptions often start with **zero GPU vCPU quota** and limited regions; request an increase early, and don't assume it will be approved. Fallbacks: Kaggle GPU (T4/P100 weekly free quota) or Colab.
- **Budget alerts:** set at $30, $60, $80 before running anything.
- **Cost discipline:** `scripts/azure_vm.ps1 start|stop|status` wraps `az vm start` / `az vm deallocate`. Always **deallocate** after a session (stopping from inside the OS does not stop billing). A deallocated VM still charges for its **disk**, so delete the VM and disk when the training work is finished.
- **Planned spend cap:** $30-40. Remaining credit is reserve for mistakes.
- **Secrets:** `.env` gitignored; no keys in code.
- Training script must be identical local vs. Azure; only `configs/azure.yaml` differs (batch size, workers).
- Download trained weights back to local before deleting the VM. The **expo runs on the local laptop only**, with no cloud dependency.

---

## 11. Tracker (copy to `docs/TRACKER.md`)

Status: `[ ]` todo, `[~]` in progress, `[x]` done with evidence. Evidence = pasted command output or file path. No evidence, no checkmark.

| Done | ID | Task | Needs | Evidence |
|---|---|---|---|---|
| [ ] | T0.1 | Repo, env, tooling | none | |
| [ ] | T0.2 | Config and logging | T0.1 | |
| [ ] | T0.3 | Decisions log | T0.1 | |
| [ ] | T1.1 | Types | T0.2 | |
| [ ] | T1.2 | Sensor emulator + frame source | T1.1 | rate measurement |
| [ ] | T1.3 | Synthetic IMU | T1.1 | |
| [ ] | T1.4 | Validation and reconnect | T1.2, T1.3 | |
| [ ] | T1.5 | Recorder and replay sources | T1.4 | |
| [ ] | T2.1 | Dataset preparation | T1.2 | split counts, histogram |
| [ ] | T2.2 | Baseline training (local) | T2.1 | val mAP |
| [ ] | T2.3 | Evaluation harness | T2.2 | `reports/eval_*.json` |
| [ ] | T2.4 | ONNX export + wrapper | T2.2 | parity + latency |
| [ ] | T2.5 | Threshold tuning | T2.3, T2.4 | sweep report |
| [ ] | T3.1 | Camera model | T1.1 | |
| [ ] | T3.2 | Rotation-delta warp | T3.1, T1.3 | |
| [ ] | T3.3 | Reprojection evaluation | T3.2, T1.2 | report table |
| [ ] | T3.4 | Latency accounting | T3.2 | |
| [ ] | T4.1 | Head-pose predictor | T1.3 | overshoot number |
| [ ] | T4.2 | Target tracker | T1.1 | |
| [ ] | T4.3 | Tracker on real detections | T4.2, T2.4 | `tracking_eval.json` |
| [ ] | T5.1 | Hot-scene handling | T1.2 | `agc_eval.md` |
| [ ] | T5.2 | Smoke simulator | T1.2 | example image |
| [ ] | T6.1 | HUD renderer | T1.1 | snapshots |
| [ ] | T6.2 | Orchestrator | T3.2, T4.1, T4.2, T6.1, T2.4 | 60 s run log |
| [ ] | T6.3 | Safety monitor + faults | T6.2 | fault test output |
| [ ] | T6.4 | Alerts + command post | T6.2 | loss test |
| [ ] | T7.1 | Expo app | T6.2, T6.3, T5.1, T5.2 | 10 min run |
| [ ] | T7.2 | Replay viewer | T1.5, T6.1 | exported clip |
| [ ] | T7.3 | Scripted scenario | T7.1 | 3 identical timelines |
| [ ] | T7.4 | Fallback video | T7.3 | mp4 |
| [ ] | T8.1 | Performance history | T6.2 | `perf_history.md` |
| [ ] | T8.2 | README, one-command run | T7.4 | |
| [ ] | T8.3 | Evaluation report | T2.5, T3.3, T4.3, T5.1 | `eval_report.md` |
| [ ] | T8.4 | Dry runs and freeze | all | tag `expo-v1` |

**Priority if time runs short (cut from the bottom):**
Must have: T0-T4, T6.1-T6.3, T7.1, T7.3, T7.4, T8.3.
Should have: T5.1, T7.2, T8.1, T8.2.
Can drop: T5.2, T6.4, T4.3.

---

## 12. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Dataset access delayed (LLVIP / FLIR ADAS require request or registration; verify) | Blocks T2 | Request on day 1; fall back to whichever is available; use KAIST as third option |
| Azure GPU quota is zero | No cloud training | Kaggle/Colab; local training at 320 input |
| Detector recall poor at small sizes | Weak headline | Report the curve honestly; it defines the realistic range, which is the credible claim |
| 8 GB RAM pressure | Crashes | Workers=2, no dataset caching, close other apps, monitor RAM in logs |
| Windows CUDA/ONNX version mismatch | Wasted days | Pin versions; CPU fallback provider; test ONNX early (T2.4) |
| Expo laptop / projector failure | Demo dies | Fallback mp4, USB copy, dry runs (T8.4) |
| Agent invents numbers or skips tests | Loss of credibility | Ground rules 2 and 3; you spot-check evidence for every task |
| Scope creep | Unfinished | Follow the priority cut list |

---

## 13. Suggested timeline (adjust to your expo date)

| Days | Work |
|---|---|
| 1-2 | T0.x, dataset requests, Azure quota check, T1.1-T1.3 |
| 3-5 | T1.4-T1.5, T2.1, start T2.2 |
| 6-8 | T2.3-T2.5, T3.1-T3.2 |
| 9-11 | T3.3-T3.4, T4.1-T4.2 |
| 12-14 | T6.1-T6.3, T5.1 |
| 15-17 | T7.1, T7.3, T7.4 |
| 18-19 | T4.3, T5.2, T7.2, T8.1 |
| 20-21 | T8.2-T8.4, rehearsal |

Add buffer of 20-30% for agent rework and debugging.

---

## 14. Things only you can do (the agent can't)

1. Request/download the thermal datasets and check licences (LLVIP, FLIR ADAS, KAIST). Verify access terms.
2. Check Azure credit, GPU quota, and set budget alerts.
3. Review each task's evidence before checking it off. Spot-run tests yourself.
4. Talk to one fire officer and get an honest quote for the pitch (not software, but it matters as much as the demo).
5. Rehearse the demo yourself, out loud, at least three times.
6. Check the expo's power, display, and time-limit rules.
