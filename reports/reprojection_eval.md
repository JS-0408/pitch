# Reprojection Quality Evaluation

Source: SyntheticThermalSource (animated gradient). IMU: SyntheticImuSource (no noise, ground truth).
Metric: mean absolute pixel error vs. ground-truth orientation frame, measured on valid-mask region.
Software-only — display hardware latency excluded.

| Profile | Delay ms | MAE no-reproj | MAE reproj | Benefit | Pass? |
|---------|----------|--------------|------------|---------|-------|
| still | 0 | 0.000 | 0.000 | +0.000 | N/A (still) |
| still | 33 | 0.000 | 0.000 | +0.000 | N/A (still) |
| still | 66 | 0.000 | 0.000 | +0.000 | N/A (still) |
| still | 100 | 0.000 | 0.000 | +0.000 | N/A (still) |
| still | 150 | 0.000 | 0.000 | +0.000 | N/A (still) |
| slow_scan | 0 | 0.000 | 0.000 | +0.000 | PASS |
| slow_scan | 33 | 4.385 | 0.000 | +4.385 | PASS |
| slow_scan | 66 | 8.457 | 0.000 | +8.457 | PASS |
| slow_scan | 100 | 12.546 | 0.000 | +12.546 | PASS |
| slow_scan | 150 | 18.382 | 0.000 | +18.382 | PASS |
| fast_turn | 0 | 0.000 | 0.000 | +0.000 | PASS |
| fast_turn | 33 | 50.265 | 0.000 | +50.265 | PASS |
| fast_turn | 66 | 84.445 | 0.000 | +84.445 | PASS |
| fast_turn | 100 | 105.194 | 0.000 | +105.194 | PASS |
| fast_turn | 150 | 116.400 | 0.000 | +116.400 | PASS |
| abrupt_reversal | 0 | 0.000 | 0.000 | +0.000 | PASS |
| abrupt_reversal | 33 | 27.893 | 0.000 | +27.893 | PASS |
| abrupt_reversal | 66 | 49.321 | 0.000 | +49.321 | PASS |
| abrupt_reversal | 100 | 53.946 | 0.000 | +53.946 | PASS |
| abrupt_reversal | 150 | 58.345 | 0.000 | +58.345 | PASS |
| walking_bob | 0 | 0.000 | 0.000 | +0.000 | PASS |
| walking_bob | 33 | 11.517 | 0.000 | +11.517 | PASS |
| walking_bob | 66 | 21.569 | 0.000 | +21.569 | PASS |
| walking_bob | 100 | 31.285 | 0.000 | +31.285 | PASS |
| walking_bob | 150 | 40.593 | 0.000 | +40.593 | PASS |

**Overall result:** All moving profiles: reprojection <= no-reprojection PASS

> Note: error values reflect synthetic animated gradient; real thermal footage may differ.