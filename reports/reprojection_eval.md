# Reprojection Quality Evaluation (Corrected)

**Method:** HeadPosePredictor-based warping vs. stale frame, measured against
ground-truth orientation. The predictor receives *noisy* IMU samples (0.5 deg/s
sigma) up to frame-capture time and must extrapolate to display time.
This is not a tautological comparison — the predictor has imperfect information.

Metric: mean absolute pixel error vs. ground-truth orientation frame,
on the valid-mask region. Synthetic animated gradient source.

| Profile | Delay ms | MAE no-reproj | MAE reproj (predicted) | Benefit | Pass? |
|---------|----------|--------------|------------------------|---------|-------|
| still | 0 | 0.000 | 0.035 | -0.035 | N/A (still) |
| still | 33 | 0.000 | 0.067 | -0.067 | N/A (still) |
| still | 66 | 0.000 | 0.106 | -0.106 | N/A (still) |
| still | 100 | 0.000 | 0.150 | -0.150 | N/A (still) |
| still | 150 | 0.000 | 0.219 | -0.219 | N/A (still) |
| slow_scan | 0 | 0.000 | 0.120 | -0.120 | REGRESSION |
| slow_scan | 33 | 4.385 | 0.267 | +4.118 | PASS |
| slow_scan | 66 | 8.450 | 0.466 | +7.984 | PASS |
| slow_scan | 100 | 12.547 | 0.759 | +11.788 | PASS |
| slow_scan | 150 | 18.383 | 1.374 | +17.009 | PASS |
| fast_turn | 0 | 0.000 | 11.286 | -11.286 | REGRESSION |
| fast_turn | 33 | 50.386 | 15.277 | +35.109 | PASS |
| fast_turn | 66 | 84.500 | 22.833 | +61.667 | PASS |
| fast_turn | 100 | 105.145 | 31.598 | +73.547 | PASS |
| fast_turn | 150 | 116.322 | 41.266 | +75.055 | PASS |
| abrupt_reversal | 0 | 0.000 | 34.041 | -34.041 | REGRESSION |
| abrupt_reversal | 33 | 23.113 | 32.869 | -9.756 | REGRESSION |
| abrupt_reversal | 66 | 41.008 | 35.028 | +5.980 | PASS |
| abrupt_reversal | 100 | 42.880 | 33.195 | +9.685 | PASS |
| abrupt_reversal | 150 | 45.986 | 44.703 | +1.283 | PASS |
| walking_bob | 0 | 0.000 | 5.577 | -5.577 | REGRESSION |
| walking_bob | 33 | 11.539 | 7.213 | +4.326 | PASS |
| walking_bob | 66 | 21.604 | 15.637 | +5.967 | PASS |
| walking_bob | 100 | 31.229 | 28.191 | +3.038 | PASS |
| walking_bob | 150 | 40.613 | 49.030 | -8.417 | REGRESSION |

**Overall:** Some regressions -- see REGRESSION rows

> Source: SyntheticThermalSource (animated gradient).
> IMU predictor receives 0.5 deg/s Gaussian noise — realistic sensor quality.
> On real textured thermal frames benefit will be larger at higher delays.