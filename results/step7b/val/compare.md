# Tracker comparison: val (stark − siamfc; D28 post-freeze extra)

A = siamfc (`/home/ruichaoz/skier_results/final/val`), B = stark (`/home/ruichaoz/skier_results/final_stark/val`). Paired 95% intervals (D25), 10,000 resamples, seed 230.

## SkiTB tracking (75 clips, 6 sequences, 12,925 frames)

| Metric | siamfc | stark | Δ | 95% interval |
| --- | --- | --- | --- | --- |
| pooled_mean_iou | 0.363 | 0.668 | +0.305 | [0.254, 0.336] |
| auc | 0.364 | 0.660 | +0.295 | [0.245, 0.326] |
| official_f_score | 0.384 | 0.681 | +0.297 | [0.250, 0.328] |

Failure episodes (IoU < 0.1 for ≥ 0.5 s): siamfc 46 in 40 clips; stark 6 in 6 clips.

## Pose under tracker crops (261 labels, 31 segments, 3 videos; seeds pooled)

Reference crops are identical for both trackers: E1 body 95.1 / ski 92.3, E2 body 95.9 / ski 93.8.

| Run | Crop | Group | siamfc | stark | Δ (points) | 95% interval |
| --- | --- | --- | --- | --- | --- | --- |
| E1 | tracked | body @0.1 | 80.2% | 95.5% | +15.4 | [8.0, 24.2] |
| E1 | tracked | ski @0.1 | 73.0% | 85.9% | +12.9 | [7.5, 20.5] |
| E1 | tracked | ski @0.05 | 62.7% | 76.4% | +13.8 | [8.9, 20.4] |
| E1 | refined | body @0.1 | 84.0% | 95.2% | +11.3 | [4.1, 19.3] |
| E1 | refined | ski @0.1 | 80.0% | 89.4% | +9.4 | [4.0, 16.0] |
| E1 | refined | ski @0.05 | 71.5% | 79.7% | +8.2 | [3.4, 13.6] |
| E2 | tracked | body @0.1 | 81.1% | 95.2% | +14.1 | [6.4, 23.4] |
| E2 | tracked | ski @0.1 | 75.4% | 88.0% | +12.6 | [7.1, 20.2] |
| E2 | tracked | ski @0.05 | 65.6% | 78.2% | +12.6 | [7.3, 19.8] |
| E2 | refined | body @0.1 | 83.9% | 94.7% | +10.7 | [3.1, 19.7] |
| E2 | refined | ski @0.1 | 81.3% | 90.9% | +9.6 | [3.8, 17.1] |
| E2 | refined | ski @0.05 | 72.7% | 81.5% | +8.8 | [3.5, 15.5] |

Labels by tracker IoU vs. the reference box: iou>=0.5 99 → 232; 0.1<=iou<0.5 129 → 27; iou<0.1 33 → 2 (siamfc → stark).

## Replay (smoothed)

SkiTB, 75 clips (mean over clips):

| Metric | siamfc | stark | Δ | 95% interval |
| --- | --- | --- | --- | --- |
| box_containment_mean | 0.943 | 0.992 | +0.049 | [0.018, 0.074] |
| frac_fully_inside | 0.892 | 0.978 | +0.086 | [0.039, 0.123] |
| area_fraction_mean | 0.457 | 0.420 | -0.037 | [-0.064, 0.004] |
| acceleration_mean | 1.823 | 1.639 | -0.184 | [-0.652, 0.223] |

Ski 2DPose, 32 segments (landmarks inside the replay crop):

| Landmarks | Labels | siamfc | stark | Δ (points) | 95% interval |
| --- | --- | --- | --- | --- | --- |
| ski_endpoints | 839 | 99.4% | 100.0% | +0.6 | [0.0, 1.6] |
| body_ski | 5511 | 99.5% | 100.0% | +0.5 | [0.0, 1.2] |
