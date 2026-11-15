# Report tables: test (stark tracks; D28 post-freeze extra)

Intervals: 95% cluster bootstrap (D25): segments within videos for pose and propagation, sequences for SkiTB; 10,000 resamples, seed 230. E1 and E2 are pooled over seeds 230 and 231 (per-seed values in brackets).

## Tracking (SkiTB test: 41 sequences, 471 SC clips, 87753 scored frames)

| Metric | Value | 95% interval |
| --- | --- | --- |
| Pooled mean IoU (ours) | 0.679 | [0.664, 0.694] |
| Per-clip mean IoU (ours) | 0.674 | |
| Success AUC (ours) | 0.670 | [0.656, 0.685] |
| Success @ IoU 0.5 (ours) | 0.794 | |
| Failure episodes (ours; IoU < 0.1 for ≥ 0.5 s) | 66 in 61 clips | |
| **Toolkit** F-score (SC) | 0.677 | [0.661, 0.693] |
| **Toolkit** precision / recall | 0.677 / 0.678 | |
| **Toolkit** GSR | 0.708 | |

## Replay (τ 0.2 s, k 2.5; low-confidence frames drive the replay, D23/D24)

| Variant | GT-box containment | Fully inside | Centering (mean / p95) | Acceleration (mean / p95) | Crop area | Ski ends inside (Ski 2DPose) |
| --- | --- | --- | --- | --- | --- | --- |
| smoothed | 0.990 [0.986, 0.994] | 0.984 | 0.081 / 0.194 | 1.69 / 7.41 | 0.43 | 99.8% (1038/1040) [99.4, 100.0] |
| unsmoothed_baseline | 0.987 [0.983, 0.992] | 0.981 | 0.056 / 0.168 | 6.57 / 25.61 | 0.36 | 99.6% (1036/1040) [99.0, 100.0] |

## Propagation: reference vs. tracker crop (k_pose 1.1; 317 labels in 40 segments)

| Run | Crop | Body @0.1 | Ski @0.1 | Ski @0.05 |
| --- | --- | --- | --- | --- |
| E1 | reference | 97.1% (7976/8210) | 94.5% (3672/3886) | 90.4% (3513/3886) |
| E1 | refined | 97.0% (7963/8210) | 91.3% (3549/3886) | 85.2% (3310/3886) |
| E1 | tracked | 96.8% (7944/8210) | 85.1% (3306/3886) | 79.0% (3068/3886) |
| E2 | reference | 97.1% (7969/8210) | 95.7% (3718/3886) | 92.1% (3578/3886) |
| E2 | refined | 96.4% (7913/8210) | 92.1% (3580/3886) | 86.6% (3365/3886) |
| E2 | tracked | 96.5% (7924/8210) | 86.5% (3362/3886) | 80.7% (3137/3886) |

Tracker-crop cut-off share: body 0.7%, ski 12.5%.

Refined crop (D26: s_min 0.3, k_refine 1.3): cut-off share body 0.2%, ski 5.0%; kept the tracker crop (fallback): E1-s230 0.0% / E1-s231 0.0% / E2-s230 0.0% / E2-s231 0.0%.

| Δ | Group | Δ (points) | 95% interval |
| --- | --- | --- | --- |
| E1 reference − E1 tracked | body @0.1 | +0.4 | [-0.4, 1.3] |
| E1 reference − E1 tracked | ski @0.1 | +9.4 | [5.3, 15.5] |
| E1 reference − E1 tracked | ski @0.05 | +11.5 | [6.6, 17.9] |
| E2 reference − E2 tracked | body @0.1 | +0.5 | [-0.3, 1.5] |
| E2 reference − E2 tracked | ski @0.1 | +9.2 | [4.9, 15.2] |
| E2 reference − E2 tracked | ski @0.05 | +11.3 | [6.6, 17.7] |
| E2 tracked − E1 tracked | body @0.1 | -0.2 | [-1.2, 0.6] |
| E2 tracked − E1 tracked | ski @0.1 | +1.4 | [0.6, 2.4] |
| E1 refined − E1 tracked | body @0.1 | +0.2 | [-0.6, 1.1] |
| E1 refined − E1 tracked | ski @0.1 | +6.3 | [3.5, 9.6] |
| E1 refined − E1 tracked | ski @0.05 | +6.2 | [3.6, 9.0] |
| E2 refined − E2 tracked | body @0.1 | -0.1 | [-0.9, 0.7] |
| E2 refined − E2 tracked | ski @0.1 | +5.6 | [2.7, 9.0] |
| E2 refined − E2 tracked | ski @0.05 | +5.9 | [3.2, 8.6] |
| E2 refined − E1 refined | body @0.1 | -0.6 | [-1.5, 0.2] |
| E2 refined − E1 refined | ski @0.1 | +0.8 | [-0.1, 1.7] |

| Stratum | Value | Run | Labels | Body ref → tracked → refined | Ski ref → tracked → refined |
| --- | --- | --- | --- | --- | --- |
| tracker_iou | 0.1<=iou<0.5 | E1 | 55 | 95.6 → 93.4 → 95.5 | 92.3 → 50.5 → 74.0 |
| tracker_iou | 0.1<=iou<0.5 | E2 | 55 | 95.4 → 93.7 → 95.5 | 92.2 → 52.2 → 74.8 |
| tracker_iou | iou>=0.5 | E1 | 262 | 97.4 → 97.4 → 97.3 | 94.9 → 91.4 → 94.5 |
| tracker_iou | iou>=0.5 | E2 | 262 | 97.4 → 97.1 → 96.6 | 96.3 → 92.8 → 95.3 |
| time_since_init | 2-5s | E1 | 116 | 98.9 → 98.2 → 98.8 | 97.1 → 88.1 → 94.5 |
| time_since_init | 2-5s | E2 | 116 | 99.1 → 98.3 → 98.3 | 97.4 → 88.4 → 94.3 |
| time_since_init | <2s | E1 | 107 | 93.8 → 94.1 → 93.5 | 90.4 → 86.8 → 89.8 |
| time_since_init | <2s | E2 | 107 | 93.2 → 93.1 → 92.1 | 92.4 → 89.7 → 90.8 |
| time_since_init | >=5s | E1 | 94 | 98.7 → 97.9 → 98.6 | 96.4 → 79.0 → 89.1 |
| time_since_init | >=5s | E2 | 94 | 98.8 → 98.1 → 98.7 | 97.6 → 79.9 → 91.1 |
| status | low_conf | E1 | 44 | 94.1 → 94.8 → 94.3 | 91.7 → 78.3 → 86.6 |
| status | low_conf | E2 | 44 | 94.4 → 93.5 → 93.7 | 92.8 → 80.0 → 88.5 |
| status | ok | E1 | 273 | 97.6 → 97.1 → 97.4 | 95.0 → 86.2 → 92.1 |
| status | ok | E2 | 273 | 97.5 → 97.0 → 96.8 | 96.2 → 87.6 → 92.7 |

