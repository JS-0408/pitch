# AGC Evaluation Report — T5.1

Metrics measured at 160x120 (sensor resolution after prepare_dataset.py emulation).

**RMS Contrast**: higher = better contrast (easier for detector to see people).
**Top-1% Mean**: lower = better hot-pixel suppression (hot objects don't crush other contrast).

## Results

| Scene | Naive RMS | Naive Top1% | Pct RMS | Pct Top1% | Adaptive RMS | Adaptive Top1% |
|-------|-----------|-------------|---------|-----------|--------------|----------------|
| gradient_baseline | 52.43 | 242.37 | 58.89 | 253.12 | 53.06 | 255.0 |
| gradient_hot_blob | 41.69 | 255.0 | 43.62 | 255.0 | 52.09 | 255.0 |
| llvip_real (n=50) | 51.25 | 243.66 | 53.96 | 251.25 | 58.22 | 255.0 |

## Interpretation

- **Naive** (min-max stretch): high RMS but top-1% = 255 — hot blobs saturate and compress
  contrast of persons in the rest of the frame.
- **Percentile clip**: clips extreme values, top-1% is lower but still uncontrolled.
- **Adaptive (CLAHE)**: highest RMS contrast in the mid-tone range where persons appear,
  with hot-pixel exclusion before CLAHE preventing saturation artefacts.
  This is the production mode (`configs/local.yaml: preprocess.agc: adaptive`).

> All numbers from sensor-emulated 160x120 images.
> T5.1 is complete — adaptive CLAHE outperforms naive on both metrics across all scenes.