# Yaazhi — Thermal Perception Pipeline

A real-time software pipeline for detecting and tracking people in low-resolution thermal video. Designed to run fully offline on a standard laptop with no physical sensors required.

---

## What It Does

Raw thermal cameras used in low-cost headsets produce video at 9 frames per second and 160×120 pixels. At those specs, a person 20 metres away is roughly 12 pixels tall. Out of the box, this video is too slow, too low-contrast, and too unstable under head movement to be useful for overlaying detection markers.

Yaazhi solves four specific problems in software:

1. **Detection** — A YOLOv8n model, trained and evaluated at actual sensor resolution (160×120), detects people in each thermal frame.
2. **Display smoothness** — The render loop runs at 60 fps regardless of the 9 Hz sensor rate. The display never blocks on inference.
3. **Marker stabilisation** — An IMU stream (real or synthetic) runs at 200 Hz. Between sensor frames, the pipeline predicts head orientation and warps the display image to compensate for head movement, keeping markers aligned.
4. **Safe failure** — A state machine monitors every input in real time. If thermal goes stale, markers fade. If IMU drops, reprojection stops. If both fail, the overlay is cleared entirely and the screen shows `SENSOR LOST. DIRECT VIEW.`

---

## Measured Performance

All numbers below are from running the evaluation scripts on this hardware:
`Ryzen 5 5600 · GTX 1650 4 GB · 8 GB RAM · Windows · Python 3.13`

| Metric | Measured value |
|---|---|
| Detection dataset | LLVIP Infrared, 3,463 test images at 160×120 sensor resolution |
| mAP50 (sensor resolution) | 61.4% |
| Recall — under 7 m | 95.7% |
| Recall — 7–15 m | 82.2% |
| Recall — 15–30 m | 70.7% |
| Recall — beyond 30 m | 56.9% |
| Inference latency (CPU) | ~2 ms per frame |
| Reprojection benefit — slow scan @ 150ms delay | +17 px MAE improvement |
| Reprojection benefit — fast turn @ 150ms delay | +75 px MAE improvement |
| Adaptive AGC vs naive RMS contrast (real frames) | 58.2 vs 51.3 |
| Render fps (target) | 60 fps |
| Unit tests | **80 / 80 passing** |

> Numbers are from `reports/`. Re-run with the evaluation scripts to reproduce.

---

## What Is Simulated

The demo runs entirely without physical hardware:

- **Thermal frames** — Public LLVIP dataset images, downscaled to 160×120, with optical blur (σ=0.8) and thermal noise (2 DN) applied to emulate a Lepton-class sensor.
- **IMU data** — Synthetic quaternion stream at 200 Hz with configurable motion profiles and gyroscope noise.
- **Smoke** — Density-controlled haze applied optionally to the visible channel. Labelled `SIMULATED SMOKE` on screen.

All simulation is labelled on screen during the demo.

---

## Repository Layout

```
yaazhi/
├── configs/
│   └── local.yaml              # All tunable parameters
├── src/yaazhi/
│   ├── types.py                # Core data contracts
│   ├── config.py               # YAML config loader
│   ├── ingestion/              # Sensor emulator, IMU synthesiser, recorder
│   ├── perception/             # Dataset prep, ONNX detector, AGC preprocessing
│   ├── reprojection/           # Camera model, rotation-delta warp
│   ├── tracking/               # Head-pose predictor, Kalman target tracker
│   ├── rendering/              # HUD renderer, ironbow palette
│   ├── safety/                 # State machine, fault injectors
│   ├── pipeline/               # Orchestrator, latency accounting
│   └── app/                    # Expo app, replay viewer
├── scripts/
│   ├── run_demo.py             # Launch the interactive demo
│   ├── evaluate.py             # Run detection evaluation
│   ├── eval_reprojection.py    # Run reprojection evaluation
│   ├── eval_tracking.py        # Run tracking evaluation
│   ├── generate_agc_report.py  # Run contrast evaluation
│   ├── train_sensor_res.py     # Fine-tune model at sensor resolution
│   ├── export_onnx.py          # Export trained model to ONNX
│   └── tune_thresholds.py      # Sweep confidence / IoU thresholds
├── tests/unit/                 # 80 unit tests
├── reports/                    # All evaluation outputs (JSON + Markdown)
├── models/                     # ONNX model file (gitignored large files)
└── docs/
    ├── TRACKER.md              # Task completion tracker
    ├── decisions.md            # Architecture decision log
    └── perf_history.md         # Per-run performance history
```

---

## Setup

**Requirements:** Python 3.10–3.13, CUDA-capable GPU recommended (CPU fallback works).

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install PyTorch with CUDA 12.4 (GTX 1650 / RTX series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Verify GPU is available
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

> For CPU-only machines, onnxruntime-gpu will fall back to CPU automatically. No code changes needed.

---

## Running the Demo

```powershell
python scripts/run_demo.py
```

**Optional flags:**
```powershell
python scripts/run_demo.py --scale 4       # Display scale factor (default: 4)
python scripts/run_demo.py --fps 60        # Render target FPS (default: 60)
python scripts/run_demo.py --duration 180  # Auto-close after N seconds
```

### Demo Controls

| Key | Action |
|---|---|
| `R` | Toggle motion reprojection ON / OFF |
| `S` | Toggle simulated smoke |
| `H` | Cycle AGC mode: Adaptive → Naive → Percentile |
| `F` | Cycle fault injection: IMU drop → Thermal freeze → Latency spike → Off |
| `1` | IMU profile: Still |
| `2` | IMU profile: Slow scan (30°/s) |
| `3` | IMU profile: Fast turn (200°/s) |
| `4` | IMU profile: Abrupt reversal |
| `5` | IMU profile: Walking bob |
| `Space` | Pause / Resume |
| `C` | Toggle session recording |
| `Q` | Quit |

---

## Running the Tests

```powershell
python -m pytest tests/ -v
```

Expected output:
```
collected 80 items

tests\unit\test_camera.py        ...   PASSED
tests\unit\test_config.py        ......PASSED
tests\unit\test_expo_app.py      .     PASSED
tests\unit\test_head_pose.py     ...   PASSED
tests\unit\test_hud.py           ......PASSED
tests\unit\test_imu_synth.py     ......PASSED
tests\unit\test_latency.py       ...   PASSED
tests\unit\test_onnx_detector.py ...   PASSED
tests\unit\test_orchestrator.py  ....  PASSED
tests\unit\test_preprocess.py    ......PASSED
tests\unit\test_recorder.py      ...   PASSED
tests\unit\test_safety.py        ....  PASSED
tests\unit\test_sources.py       ....  PASSED
tests\unit\test_tracker.py       ..... PASSED
tests\unit\test_types.py         ......PASSED
tests\unit\test_validation.py    ......PASSED
tests\unit\test_warp.py          ...   PASSED

80 passed in ~8s
```

---

## Running Individual Evaluations

### Detection (requires model + dataset)
```powershell
python scripts/eval_detector.py --model models/yolov8n_llvip.onnx
# Output → reports/eval_yolov8n_llvip.json
#          reports/eval_recall_by_bucket.md
```

### Reprojection quality
```powershell
python scripts/eval_reprojection.py
# Output → reports/reprojection_eval.md
```

### Tracker evaluation
```powershell
python scripts/eval_tracking.py
# Output → reports/tracking_eval.json
```

### Contrast / AGC comparison
```powershell
python scripts/generate_agc_report.py
# Output → reports/agc_eval.md
```

### Threshold sweep
```powershell
python scripts/tune_thresholds.py
# Output → reports/threshold_sweep.md
```

---

## Configuration

All parameters are in `configs/local.yaml`. No values are hardcoded in module code.

Key parameters:

```yaml
sensor:
  width: 160          # Emulated sensor resolution
  height: 120
  fps: 9              # Sensor frame rate

display:
  render_fps: 60      # HUD render rate (independent of sensor)

detector:
  conf: 0.10          # Detection confidence threshold (tuned by sweep)
  nms_iou: 0.45       # NMS IoU threshold

reprojection:
  enabled: true
  max_delta_deg: 60   # Clamp rotation delta beyond this

safety:
  frame_stale_ms: 400   # Thermal timeout before DEGRADED
  imu_stale_ms: 100     # IMU timeout before reprojection stop
  safe_after_ms: 1200   # Time to SAFE state
```

---

## Known Limitations

- Detection recall drops to ~57% beyond 30 metres on a 160×120 sensor. This is a hardware physics constraint, not a software bug.
- Abrupt head reversals cause a brief reprojection overshoot before the predictor corrects. Documented in `reports/reprojection_eval.md`.
- The IMU in the demo is synthetic. Real headset integration requires a USB/Ethernet source implementing the `ImuSource` protocol in `src/yaazhi/types.py`.
- No performance measurement has been done on embedded compute (Jetson class). Current baselines are from a GTX 1650 desktop GPU.
- No regulatory certification of any kind. This is a research and development prototype.

---

## Hardware Integration Path

To connect a real thermal camera and IMU, implement two protocols defined in `src/yaazhi/types.py`:

```python
class FrameSource(Protocol):
    def read(self) -> ThermalFrame | None: ...

class ImuSource(Protocol):
    def read(self) -> ImuSample | None: ...
```

Pass your implementations to the `Orchestrator` in place of the mock sources. Everything else — detection, tracking, reprojection, safety, rendering — remains unchanged.

---

## License

Research and development prototype. Not for operational deployment.
