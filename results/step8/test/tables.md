# Report tables: test

Intervals: 95% cluster bootstrap (D25): segments within videos for pose and propagation, sequences for SkiTB; 10,000 resamples, seed 230. E1 and E2 are pooled over seeds 230 and 231 (per-seed values in brackets).

## Pose (360 images, 43 segments, 3 videos)

| Run | Body @0.1 | Ski @0.1 | Ski @0.05 | Body, shared joints @0.1 | Poles @0.1 (suppl.) |
| --- | --- | --- | --- | --- | --- |
| E0 | N/A | N/A | N/A | 94.7% (3113/3287) | N/A |
| E1 | 96.3% (8986/9328) (96.6 / 96.1) | 94.1% (4136/4394) (94.5 / 93.8) | 89.8% (3948/4394) (90.3 / 89.4) | 96.2% (6322/6574) (96.6 / 95.7) | 4.3% (53/1228) (4.1 / 4.6) |
| E2 | 95.9% (8941/9328) (96.2 / 95.5) | 94.8% (4165/4394) (95.2 / 94.4) | 91.0% (3999/4394) (91.8 / 90.2) | 95.6% (6282/6574) (95.9 / 95.3) | 5.0% (61/1228) (4.9 / 5.0) |

| Δ | Group | Δ (points) | 95% interval |
| --- | --- | --- | --- |
| E2 − E1 | body @0.1 | -0.5 | [-1.3, 0.2] |
| E2 − E1 | ski @0.1 | +0.7 | [-0.3, 1.5] |
| E2 − E1 | ski @0.05 | +1.2 | [0.2, 2.2] |
| E2 − E1 | body_shared @0.1 | -0.6 | [-1.5, 0.2] |
| E2 − E1 | pole @0.1 | +0.7 | [-0.7, 1.9] |
| E1 − E0 | body_shared @0.1 | +1.5 | [-1.0, 3.7] |
| E1 − E0 | body_shared @0.05 | +4.9 | [2.6, 7.0] |
| E2 − E0 | body_shared @0.1 | +0.9 | [-1.1, 2.6] |
| E2 − E0 | body_shared @0.05 | +4.4 | [2.3, 6.3] |

Per video, PCK@0.1:

| Run | Group | NrWcP1s3QC0 | dpFlXMvvZ9E | uLW74013Wp0 |
| --- | --- | --- | --- | --- |
| E0 | body_shared | 83.5% (434/520) | 70.2% (87/124) | 98.1% (2592/2643) |
| E1 | body | 86.0% (1274/1482) | 84.6% (301/356) | 98.9% (7411/7490) |
| E1 | ski | 87.4% (629/720) | 64.4% (94/146) | 96.7% (3413/3528) |
| E1 | body_shared | 84.2% (876/1040) | 84.7% (210/248) | 99.1% (5236/5286) |
| E2 | body | 82.5% (1223/1482) | 84.0% (299/356) | 99.1% (7419/7490) |
| E2 | ski | 85.0% (612/720) | 62.3% (91/146) | 98.1% (3462/3528) |
| E2 | body_shared | 80.3% (835/1040) | 83.5% (207/248) | 99.1% (5240/5286) |

## Tracking (SkiTB test: 41 sequences, 471 SC clips, 87753 scored frames)

| Metric | Value | 95% interval |
| --- | --- | --- |
| Pooled mean IoU (ours) | 0.404 | [0.380, 0.428] |
| Per-clip mean IoU (ours) | 0.403 | |
| Success AUC (ours) | 0.404 | [0.380, 0.428] |
| Success @ IoU 0.5 (ours) | 0.444 | |
| Failure episodes (ours; IoU < 0.1 for ≥ 0.5 s) | 258 in 234 clips | |
| **Toolkit** F-score (SC) | 0.406 | [0.387, 0.428] |
| **Toolkit** precision / recall | 0.404 / 0.409 | |
| **Toolkit** GSR | 0.516 | |

## Replay (τ 0.2 s, k 2.5; low-confidence frames drive the replay, D23/D24)

| Variant | GT-box containment | Fully inside | Centering (mean / p95) | Acceleration (mean / p95) | Crop area | Ski ends inside (Ski 2DPose) |
| --- | --- | --- | --- | --- | --- | --- |
| smoothed | 0.963 [0.951, 0.973] | 0.935 | 0.111 / 0.253 | 1.38 / 5.41 | 0.48 | 93.7% (974/1040) [86.5, 99.4] |
| unsmoothed_baseline | 0.944 [0.931, 0.958] | 0.906 | 0.104 / 0.263 | 4.11 / 14.86 | 0.43 | 92.8% (965/1040) [84.6, 99.5] |

## Propagation: reference vs. tracker crop (k_pose 1.1; 317 labels in 40 segments)

| Run | Crop | Body @0.1 | Ski @0.1 | Ski @0.05 |
| --- | --- | --- | --- | --- |
| E1 | reference | 97.1% (7976/8210) | 94.5% (3672/3886) | 90.4% (3513/3886) |
| E1 | refined | 82.1% (6740/8210) | 75.1% (2919/3886) | 69.1% (2684/3886) |
| E1 | tracked | 79.0% (6490/8210) | 66.1% (2568/3886) | 57.9% (2250/3886) |
| E2 | reference | 97.1% (7969/8210) | 95.7% (3718/3886) | 92.1% (3578/3886) |
| E2 | refined | 82.4% (6765/8210) | 77.0% (2992/3886) | 71.0% (2760/3886) |
| E2 | tracked | 78.9% (6479/8210) | 67.7% (2631/3886) | 59.3% (2305/3886) |

Tracker-crop cut-off share: body 16.6%, ski 30.5%.

Refined crop (D26: s_min 0.3, k_refine 1.3): cut-off share body 13.4%, ski 19.9%; kept the tracker crop (fallback): E1-s230 8.2% / E1-s231 7.9% / E2-s230 9.5% / E2-s231 10.4%.

| Δ | Group | Δ (points) | 95% interval |
| --- | --- | --- | --- |
| E1 reference − E1 tracked | body @0.1 | +18.1 | [9.3, 27.8] |
| E1 reference − E1 tracked | ski @0.1 | +28.4 | [19.9, 38.0] |
| E1 reference − E1 tracked | ski @0.05 | +32.5 | [24.5, 41.0] |
| E2 reference − E2 tracked | body @0.1 | +18.1 | [9.3, 27.9] |
| E2 reference − E2 tracked | ski @0.1 | +28.0 | [19.3, 37.7] |
| E2 reference − E2 tracked | ski @0.05 | +32.8 | [24.7, 41.5] |
| E2 tracked − E1 tracked | body @0.1 | -0.1 | [-0.7, 0.4] |
| E2 tracked − E1 tracked | ski @0.1 | +1.6 | [0.8, 2.5] |
| E1 refined − E1 tracked | body @0.1 | +3.0 | [0.9, 5.6] |
| E1 refined − E1 tracked | ski @0.1 | +9.0 | [6.4, 11.9] |
| E1 refined − E1 tracked | ski @0.05 | +11.2 | [7.9, 14.4] |
| E2 refined − E2 tracked | body @0.1 | +3.5 | [1.3, 6.0] |
| E2 refined − E2 tracked | ski @0.1 | +9.3 | [6.7, 12.1] |
| E2 refined − E2 tracked | ski @0.05 | +11.7 | [8.5, 14.9] |
| E2 refined − E1 refined | body @0.1 | +0.3 | [-0.8, 1.4] |
| E2 refined − E1 refined | ski @0.1 | +1.9 | [1.0, 2.8] |

| Stratum | Value | Run | Labels | Body ref → tracked → refined | Ski ref → tracked → refined |
| --- | --- | --- | --- | --- | --- |
| tracker_iou | 0.1<=iou<0.5 | E1 | 136 | 97.4 → 89.4 → 95.0 | 95.6 → 65.9 → 81.5 |
| tracker_iou | 0.1<=iou<0.5 | E2 | 136 | 97.7 → 90.5 → 95.6 | 96.3 → 68.6 → 84.4 |
| tracker_iou | iou<0.1 | E1 | 50 | 99.0 → 5.8 → 12.7 | 97.2 → 6.7 → 13.7 |
| tracker_iou | iou<0.1 | E2 | 50 | 98.9 → 5.8 → 14.9 | 98.4 → 6.7 → 14.3 |
| tracker_iou | iou>=0.5 | E1 | 131 | 96.2 → 96.8 → 95.8 | 92.5 → 89.2 → 93.4 |
| tracker_iou | iou>=0.5 | E2 | 131 | 95.7 → 95.4 → 95.1 | 94.1 → 90.5 → 94.9 |
| time_since_init | 2-5s | E1 | 116 | 98.9 → 82.5 → 85.0 | 97.1 → 67.9 → 77.7 |
| time_since_init | 2-5s | E2 | 116 | 99.1 → 82.9 → 84.9 | 97.4 → 68.4 → 77.6 |
| time_since_init | <2s | E1 | 107 | 93.8 → 87.1 → 87.4 | 90.4 → 76.6 → 81.9 |
| time_since_init | <2s | E2 | 107 | 93.2 → 86.4 → 87.7 | 92.4 → 79.3 → 85.7 |
| time_since_init | >=5s | E1 | 94 | 98.7 → 65.8 → 72.6 | 96.4 → 50.2 → 63.1 |
| time_since_init | >=5s | E2 | 94 | 98.8 → 65.8 → 73.5 | 97.6 → 51.8 → 65.0 |
| status | low_conf | E1 | 116 | 98.9 → 72.5 → 78.2 | 97.1 → 54.6 → 67.7 |
| status | low_conf | E2 | 116 | 98.7 → 73.0 → 80.2 | 97.5 → 56.2 → 70.5 |
| status | ok | E1 | 201 | 96.1 → 82.9 → 84.3 | 93.2 → 71.8 → 78.8 |
| status | ok | E2 | 201 | 96.1 → 82.4 → 83.7 | 94.8 → 73.5 → 80.2 |

