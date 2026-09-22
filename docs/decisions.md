# Decisions Log
# T0.3 — Seeded with all major architectural decisions per spec.

## Decision Record: Yaazhi Vision Prototype

| # | Date | Decision | Reason |
|---|------|----------|--------|
| 1 | 2026-09-22 | **Windows-native development** (no WSL2) | 8 GB RAM — WSL2 overhead hurts; CUDA PyTorch runs natively; expo laptop is Windows |
| 2 | 2026-09-22 | **ONNX Runtime, not TensorRT** | TensorRT requires engine compilation per GPU; ONNX Runtime with CUDA provider is sufficient for GTX 1650 prototype latency budget (<15 ms); TensorRT is a hardware-track task |
| 3 | 2026-09-22 | **Sensor emulation at 160×120 / 9 Hz** | Real target sensor (Lepton-class) is 160×120 at ~9 Hz; all training and evaluation data downscaled to this resolution so results are relevant to the actual hardware |
| 4 | 2026-09-22 | **Render loop as fast loop (60 Hz), inference as slow loop (~9 Hz)** | Decoupling display refresh from sensor rate allows smooth, stable HUD at 60 fps using IMU extrapolation, independent of thermal frame rate |
| 5 | 2026-09-22 | **YOLOv8n at 320 input** | Smallest Ultralytics model; fits within 4 GB VRAM; inference latency budget <15 ms on GTX 1650; acceptable recall for small-person detection at emulated sensor resolution |
| 6 | 2026-09-22 | **`configs/*.yaml` for all hyperparameters** | No hardcoded values in module code; single source of truth; enables local↔azure split cleanly |
| 7 | 2026-09-22 | **Monotonic clock (`time.monotonic_ns()`) for all timestamps** | Single reference clock across all threads; prevents OS wall-clock drift from corrupting latency measurements |
| 8 | 2026-09-22 | **Structured key=value logging** | Machine-parseable; no print() in module code; grep-friendly for post-run analysis |
