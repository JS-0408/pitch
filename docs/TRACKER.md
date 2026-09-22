# Tracker

Status: `[ ]` todo, `[~]` in progress, `[x]` done with evidence. Evidence = pasted command output or file path. No evidence, no checkmark.

| Done | ID | Task | Needs | Evidence |
|---|---|---|---|---|
| [x] | T0.1 | Repo, env, tooling | none | `torch.cuda.is_available()=True`, `pytest -q` 47 passed in 4.06s; Python 3.13.13, torch 2.6.0+cu124, GTX 1650 |
| [x] | T0.2 | Config and logging | T0.1 | 6 unit tests pass (load, override, missing-key, readonly, invalid-env, env-var) |
| [x] | T0.3 | Decisions log | T0.1 | `docs/decisions.md` — 8 decisions seeded |
| [x] | T1.1 | Types | T0.2 | 12 tests pass; frozen classes reject mutation; all enums correct |
| [x] | T1.2 | Sensor emulator + frame source | T1.1 | rate within ±5% at 30 Hz test (2% at 9 Hz expected); 4 tests pass |
| [x] | T1.3 | Synthetic IMU | T1.1 | 6 tests pass; all profiles unit-norm; fast_turn peak ≥100 deg/s; ground truth deterministic |
| [x] | T1.4 | Validation and reconnect | T1.2, T1.3 | 6 tests pass; wrong-shape, non-monotonic rejection; 3-fault source recovers |
| [x] | T1.5 | Recorder and replay sources | T1.4 | 3 tests pass; frame count, timestamps, IMU sample count identical after round-trip |
| [ ] | T2.1 | Dataset preparation | T1.2 | split counts, histogram |
| [ ] | T2.2 | Baseline training (local) | T2.1 | val mAP |
| [ ] | T2.3 | Evaluation harness | T2.2 | `reports/eval_*.json` |
| [ ] | T2.4 | ONNX export + wrapper | T2.2 | parity + latency |
| [ ] | T2.5 | Threshold tuning | T2.3, T2.4 | sweep report |
| [x] | T3.1 | Camera model | T1.1 | 3 tests pass; center→optical-axis; round-trip error <1e-6; fx from FOV verified |
| [x] | T3.2 | Rotation-delta warp | T3.1, T1.3 | 3 tests pass; zero-delta identity; 10° yaw shift ≤±2px; >60° clamped |
| [x] | T3.3 | Reprojection evaluation | T3.2, T1.2 | `reports/reprojection_eval.md`; reproj MAE=0.00 vs no-reproj up to 116.4 on fast_turn@150ms; all 5 profiles PASS |
| [x] | T3.4 | Latency accounting | T3.2 | 3 tests pass; M2D=15ms, frame-age=20ms verified; rolling window cap works |
| [x] | T4.1 | Head-pose predictor | T1.3 | 3 tests pass; steady-turn err=0.000 deg at 50ms horizon; abrupt-reversal overshoot=0.00 deg (documented) |
| [x] | T4.2 | Target tracker | T1.1 | 7 tests pass; CV-RMSE=0.11px; COASTING->LOST at correct ms; single FP never CONFIRMED; predict() non-mutating |
| [ ] | T4.3 | Tracker on real detections | T4.2, T2.4 | `tracking_eval.json` |
| [x] | T5.1 | Hot-scene handling | T1.2 | 5 tests pass; naive RMS=24.8, adaptive=19.0 on hot-blob scene; `reports/agc_eval.md` via preprocess tests |
| [x] | T5.2 | Smoke simulator | T1.2 | density=0 identity confirmed; thermal/visible separate paths; SmokeSim toggleable |
| [x] | T6.1 | HUD renderer | T1.1 | 6 tests pass; ironbow LUT shape/dtype; marker pixel presence; SAFE banner; output shape 640x504 |
| [ ] | T6.2 | Orchestrator | T3.2, T4.1, T4.2, T6.1, T2.4 | 60 s run log |
| [x] | T6.3 | Safety monitor + faults | T6.2 | 4 tests pass; NORMAL→DEGRADED→SAFE; recovery requires N=10 good samples |
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
