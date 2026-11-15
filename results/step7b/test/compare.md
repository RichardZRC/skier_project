# Tracker comparison: test (stark − siamfc; D28 post-freeze extra)

A = siamfc (`/home/ruichaoz/skier_results/final/test`), B = stark (`/home/ruichaoz/skier_results/final_stark/test`). Paired 95% intervals (D25), 10,000 resamples, seed 230.

## SkiTB tracking (471 clips, 41 sequences, 87,753 frames)

| Metric | siamfc | stark | Δ | 95% interval |
| --- | --- | --- | --- | --- |
| pooled_mean_iou | 0.404 | 0.679 | +0.276 | [0.251, 0.301] |
| auc | 0.404 | 0.670 | +0.266 | [0.242, 0.291] |
| official_f_score | 0.406 | 0.677 | +0.270 | [0.246, 0.294] |

Failure episodes (IoU < 0.1 for ≥ 0.5 s): siamfc 258 in 234 clips; stark 66 in 61 clips.

## Pose under tracker crops (317 labels, 40 segments, 3 videos; seeds pooled)

Reference crops are identical for both trackers: E1 body 97.1 / ski 94.5, E2 body 97.1 / ski 95.7.

| Run | Crop | Group | siamfc | stark | Δ (points) | 95% interval |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | tracked | body @0.1 | 79.0% | 96.8% | +17.7 | [9.3, 27.1] |
| E1 | tracked | ski @0.1 | 66.1% | 85.1% | +19.0 | [11.5, 27.6] |
| E1 | tracked | ski @0.05 | 57.9% | 79.0% | +21.0 | [14.0, 28.6] |
| E1 | refined | body @0.1 | 82.1% | 97.0% | +14.9 | [6.3, 24.7] |
| E1 | refined | ski @0.1 | 75.1% | 91.3% | +16.2 | [8.7, 25.3] |
| E1 | refined | ski @0.05 | 69.1% | 85.2% | +16.1 | [9.2, 24.1] |
| E2 | tracked | body @0.1 | 78.9% | 96.5% | +17.6 | [9.1, 27.0] |
| E2 | tracked | ski @0.1 | 67.7% | 86.5% | +18.8 | [11.1, 27.8] |
| E2 | tracked | ski @0.05 | 59.3% | 80.7% | +21.4 | [14.0, 29.5] |
| E2 | refined | body @0.1 | 82.4% | 96.4% | +14.0 | [5.8, 23.4] |
| E2 | refined | ski @0.1 | 77.0% | 92.1% | +15.1 | [7.6, 24.1] |
| E2 | refined | ski @0.05 | 71.0% | 86.6% | +15.6 | [8.7, 23.5] |

Labels by tracker IoU vs. the reference box: iou>=0.5 131 → 262; 0.1<=iou<0.5 136 → 55; iou<0.1 50 → 0 (siamfc → stark).

## Replay (smoothed)

SkiTB, 471 clips (mean over clips):

| Metric | siamfc | stark | Δ | 95% interval |
| --- | --- | --- | --- | --- |
| box_containment_mean | 0.963 | 0.990 | +0.027 | [0.016, 0.040] |
| frac_fully_inside | 0.935 | 0.984 | +0.049 | [0.036, 0.064] |
| area_fraction_mean | 0.483 | 0.427 | -0.056 | [-0.075, -0.034] |
| acceleration_mean | 1.382 | 1.692 | +0.310 | [0.137, 0.486] |

Ski 2DPose, 43 segments (landmarks inside the replay crop):

| Landmarks | Labels | siamfc | stark | Δ (points) | 95% interval |
| --- | --- | --- | --- | --- | --- |
| ski_endpoints | 1040 | 93.7% | 99.8% | +6.2 | [0.6, 13.2] |
| body_ski | 6861 | 95.2% | 99.9% | +4.7 | [0.3, 10.3] |
