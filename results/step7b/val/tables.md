# Report tables: val (stark tracks; D28 post-freeze extra)

Intervals: 95% cluster bootstrap (D25): segments within videos for pose and propagation, sequences for SkiTB; 10,000 resamples, seed 230. E1 and E2 are pooled over seeds 230 and 231 (per-seed values in brackets).

## Tracking (SkiTB val: 6 sequences, 75 SC clips, 12925 scored frames)

| Metric | Value | 95% interval |
| --- | --- | --- |
| Pooled mean IoU (ours) | 0.668 | [0.628, 0.719] |
| Per-clip mean IoU (ours) | 0.675 | |
| Success AUC (ours) | 0.660 | [0.622, 0.709] |
| Success @ IoU 0.5 (ours) | 0.774 | |
| Failure episodes (ours; IoU < 0.1 for ≥ 0.5 s) | 6 in 6 clips | |
| **Toolkit** F-score (SC) | 0.681 | [0.647, 0.719] |
| **Toolkit** precision / recall | 0.686 / 0.677 | |
| **Toolkit** GSR | 0.718 | |

## Replay (τ 0.2 s, k 2.5; low-confidence frames drive the replay, D23/D24)

| Variant | GT-box containment | Fully inside | Centering (mean / p95) | Acceleration (mean / p95) | Crop area | Ski ends inside (Ski 2DPose) |
| --- | --- | --- | --- | --- | --- | --- |
| smoothed | 0.992 [0.981, 1.000] | 0.978 | 0.084 / 0.194 | 1.64 / 6.68 | 0.42 | 100.0% (839/839) [100.0, 100.0] |
| unsmoothed_baseline | 0.988 [0.974, 0.999] | 0.975 | 0.055 / 0.161 | 6.66 / 24.77 | 0.35 | 100.0% (839/839) [100.0, 100.0] |

## Propagation: reference vs. tracker crop (k_pose 1.1; 261 labels in 31 segments)

| Run | Crop | Body @0.1 | Ski @0.1 | Ski @0.05 |
| --- | --- | --- | --- | --- |
| E1 | reference | 95.1% (6387/6716) | 92.3% (2867/3106) | 83.6% (2597/3106) |
| E1 | refined | 95.2% (6395/6716) | 89.4% (2778/3106) | 79.7% (2477/3106) |
| E1 | tracked | 95.5% (6417/6716) | 85.9% (2668/3106) | 76.4% (2374/3106) |
| E2 | reference | 95.9% (6440/6716) | 93.8% (2913/3106) | 85.2% (2647/3106) |
| E2 | refined | 94.7% (6357/6716) | 90.9% (2823/3106) | 81.5% (2530/3106) |
| E2 | tracked | 95.2% (6396/6716) | 88.0% (2732/3106) | 78.2% (2429/3106) |

Tracker-crop cut-off share: body 1.1%, ski 9.6%.

Refined crop (D26: s_min 0.3, k_refine 1.3): cut-off share body 0.7%, ski 3.9%; kept the tracker crop (fallback): E1-s230 0.0% / E1-s231 0.4% / E2-s230 0.8% / E2-s231 0.0%.

| Δ | Group | Δ (points) | 95% interval |
| --- | --- | --- | --- |
| E1 reference − E1 tracked | body @0.1 | -0.4 | [-2.0, 1.6] |
| E1 reference − E1 tracked | ski @0.1 | +6.4 | [3.3, 9.1] |
| E1 reference − E1 tracked | ski @0.05 | +7.2 | [4.1, 9.7] |
| E2 reference − E2 tracked | body @0.1 | +0.7 | [-0.9, 2.9] |
| E2 reference − E2 tracked | ski @0.1 | +5.8 | [2.7, 8.6] |
| E2 reference − E2 tracked | ski @0.05 | +7.0 | [3.7, 9.9] |
| E2 tracked − E1 tracked | body @0.1 | -0.3 | [-1.0, 0.4] |
| E2 tracked − E1 tracked | ski @0.1 | +2.1 | [1.0, 3.2] |
| E1 refined − E1 tracked | body @0.1 | -0.3 | [-0.9, 0.3] |
| E1 refined − E1 tracked | ski @0.1 | +3.5 | [1.9, 4.8] |
| E1 refined − E1 tracked | ski @0.05 | +3.3 | [1.1, 5.1] |
| E2 refined − E2 tracked | body @0.1 | -0.6 | [-1.4, 0.1] |
| E2 refined − E2 tracked | ski @0.1 | +2.9 | [1.3, 4.4] |
| E2 refined − E2 tracked | ski @0.05 | +3.3 | [0.9, 5.3] |
| E2 refined − E1 refined | body @0.1 | -0.6 | [-1.7, 0.4] |
| E2 refined − E1 refined | ski @0.1 | +1.4 | [0.4, 2.7] |

| Stratum | Value | Run | Labels | Body ref → tracked → refined | Ski ref → tracked → refined |
| --- | --- | --- | --- | --- | --- |
| tracker_iou | 0.1<=iou<0.5 | E1 | 27 | 95.3 → 95.3 → 96.7 | 96.3 → 71.5 → 85.9 |
| tracker_iou | 0.1<=iou<0.5 | E2 | 27 | 96.4 → 95.8 → 97.3 | 94.8 → 74.1 → 87.8 |
| tracker_iou | iou<0.1 | E1 | 2 | 100.0 → 0.0 → 0.0 | 95.8 → 0.0 → 0.0 |
| tracker_iou | iou<0.1 | E2 | 2 | 100.0 → 0.0 → 0.0 | 91.7 → 0.0 → 0.0 |
| tracker_iou | iou>=0.5 | E1 | 232 | 95.0 → 96.3 → 95.8 | 91.9 → 88.0 → 90.5 |
| tracker_iou | iou>=0.5 | E2 | 232 | 95.8 → 95.9 → 95.1 | 93.7 → 90.0 → 92.0 |
| time_since_init | 2-5s | E1 | 120 | 96.6 → 95.4 → 95.5 | 95.1 → 87.9 → 91.6 |
| time_since_init | 2-5s | E2 | 120 | 97.3 → 95.6 → 95.8 | 95.9 → 90.5 → 92.6 |
| time_since_init | <2s | E1 | 104 | 93.3 → 95.1 → 94.3 | 88.0 → 85.6 → 87.9 |
| time_since_init | <2s | E2 | 104 | 94.1 → 94.3 → 92.9 | 90.4 → 87.7 → 89.9 |
| time_since_init | >=5s | E1 | 37 | 95.5 → 97.3 → 97.1 | 96.5 → 79.9 → 86.7 |
| time_since_init | >=5s | E2 | 37 | 96.4 → 96.8 → 95.9 | 97.5 → 79.9 → 88.2 |
| status | low_conf | E1 | 76 | 95.4 → 95.6 → 95.0 | 94.4 → 80.4 → 87.7 |
| status | low_conf | E2 | 76 | 95.9 → 95.1 → 95.5 | 95.3 → 82.9 → 88.3 |
| status | ok | E1 | 185 | 95.0 → 95.5 → 95.3 | 91.5 → 88.1 → 90.1 |
| status | ok | E2 | 185 | 95.9 → 95.3 → 94.3 | 93.2 → 90.0 → 91.9 |

