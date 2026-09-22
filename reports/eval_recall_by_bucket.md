# Recall by Person Height Bucket — Sensor-Resolution (160x120)

Person height measured at sensor resolution (160x120). Aggregate recall
hides the large performance gap between close and far targets.
Use this table in pitch materials, not the 640px aggregate.

| Height bucket | Meaning (approx distance) | TP | FN | Recall |
|---------------|--------------------------|----|----|--------|
| 4-8px (far) | >30 m | 82 | 62 | 56.9% |
| 8-15px (mid) | 15–30 m | 350 | 145 | 70.7% |
| 15-30px (close) | 7–15 m | 3220 | 698 | 82.2% |
| >30px (near) | <7 m | 3576 | 161 | 95.7% |

**Overall sensor-res mAP50:** 61.4% @ 160x120 (conf=0.10, iou=0.45)

> Note: model was trained at 640px. Fine-tuning at 160px will improve
> recall in the 4-15px buckets (far targets). This table is the honest
> baseline before any sensor-resolution fine-tuning.