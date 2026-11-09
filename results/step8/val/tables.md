# Report tables: val

Intervals: 95% cluster bootstrap (D25): segments within videos for pose and propagation, sequences for SkiTB; 10,000 resamples, seed 230. E1 and E2 are pooled over seeds 230 and 231 (per-seed values in brackets).

## Pose (293 images, 32 segments, 3 videos)

| Run | Body @0.1 | Ski @0.1 | Ski @0.05 | Body, shared joints @0.1 | Poles @0.1 (suppl.) |
| --- | --- | --- | --- | --- | --- |
| E0 | N/A | N/A | N/A | 95.1% (2517/2647) | N/A |
| E1 | 95.2% (7155/7518) (95.6 / 94.8) | 91.9% (3220/3504) (92.2 / 91.6) | 82.9% (2904/3504) (83.1 / 82.6) | 95.5% (5054/5294) (95.8 / 95.2) | 5.3% (49/932) (5.4 / 5.2) |
| E2 | 95.5% (7178/7518) (95.1 / 95.9) | 93.1% (3263/3504) (92.5 / 93.7) | 84.3% (2955/3504) (84.4 / 84.3) | 95.8% (5070/5294) (95.2 / 96.3) | 6.0% (56/932) (6.4 / 5.6) |

| Δ | Group | Δ (points) | 95% interval |
| --- | --- | --- | --- |
| E2 − E1 | body @0.1 | +0.3 | [-0.7, 1.2] |
| E2 − E1 | ski @0.1 | +1.2 | [0.1, 2.3] |
| E2 − E1 | ski @0.05 | +1.5 | [0.4, 2.6] |
| E2 − E1 | body_shared @0.1 | +0.3 | [-0.8, 1.3] |
| E2 − E1 | pole @0.1 | +0.8 | [-0.9, 2.2] |
| E1 − E0 | body_shared @0.1 | +0.4 | [-1.2, 1.7] |
| E1 − E0 | body_shared @0.05 | +4.6 | [2.4, 6.7] |
| E2 − E0 | body_shared @0.1 | +0.7 | [-1.0, 2.1] |
| E2 − E0 | body_shared @0.05 | +4.6 | [2.5, 6.6] |

Per video, PCK@0.1:

| Run | Group | SVYdH614G_M | qxfgw1Kd98A | v7PADD2_I00 |
| --- | --- | --- | --- | --- |
| E0 | body_shared | 94.0% (328/349) | 95.3% (2071/2174) | 95.2% (118/124) |
| E1 | body | 92.5% (908/982) | 95.7% (5907/6172) | 93.4% (340/364) |
| E1 | ski | 80.0% (416/520) | 94.6% (2638/2788) | 84.7% (166/196) |
| E1 | body_shared | 93.8% (655/698) | 95.7% (4162/4348) | 95.6% (237/248) |
| E2 | body | 92.1% (904/982) | 96.1% (5930/6172) | 94.5% (344/364) |
| E2 | ski | 84.2% (438/520) | 95.5% (2662/2788) | 83.2% (163/196) |
| E2 | body_shared | 93.6% (653/698) | 96.1% (4178/4348) | 96.4% (239/248) |

## Augmentation study (val, D16/D21)

| Run | Crops | Body @0.1 | Ski @0.1 | Ski @0.05 |
| --- | --- | --- | --- | --- |
| E1 | clean | 95.2% | 91.9% (92.2 / 91.6) | 82.9% |
| E2 | clean | 95.5% | 93.1% (92.5 / 93.7) | 84.3% |
| E3 | clean | 96.9% | 92.0% (91.8 / 92.3) | 82.0% |
| E4 | clean | 97.1% | 91.0% (90.8 / 91.1) | 78.8% |
| E1 | moderate | 95.2% | 90.4% (90.5 / 90.4) | 80.9% |
| E2 | moderate | 95.3% | 92.9% (92.3 / 93.4) | 83.5% |
| E3 | moderate | 96.6% | 90.9% (90.6 / 91.2) | 80.5% |
| E4 | moderate | 96.5% | 89.5% (89.3 / 89.7) | 78.7% |

| Δ vs. E1 | Crops | Group | Δ (points) | 95% interval |
| --- | --- | --- | --- | --- |
| E2 | clean | body | +0.3 | [-0.7, 1.2] |
| E2 | clean | ski | +1.2 | [0.1, 2.3] |
| E3 | clean | body | +1.7 | [0.7, 2.7] |
| E3 | clean | ski | +0.1 | [-1.1, 1.5] |
| E4 | clean | body | +1.9 | [0.9, 3.0] |
| E4 | clean | ski | -0.9 | [-2.0, 0.1] |
| E2 | moderate | body | +0.2 | [-0.9, 1.2] |
| E2 | moderate | ski | +2.5 | [1.5, 3.6] |
| E3 | moderate | body | +1.4 | [0.7, 2.4] |
| E3 | moderate | ski | +0.5 | [-0.9, 2.1] |
| E4 | moderate | body | +1.4 | [0.6, 2.3] |
| E4 | moderate | ski | -0.9 | [-2.4, 0.9] |

## Tracking (SkiTB val: 6 sequences, 75 SC clips, 12925 scored frames)

| Metric | Value | 95% interval |
| --- | --- | --- |
| Pooled mean IoU (ours) | 0.363 | [0.299, 0.453] |
| Per-clip mean IoU (ours) | 0.378 | |
| Success AUC (ours) | 0.364 | [0.301, 0.453] |
| Success @ IoU 0.5 (ours) | 0.372 | |
| Failure episodes (ours; IoU < 0.1 for ≥ 0.5 s) | 46 in 40 clips | |
| **Toolkit** F-score (SC) | 0.384 | [0.324, 0.463] |
| **Toolkit** precision / recall | 0.381 / 0.387 | |
| **Toolkit** GSR | 0.485 | |

## Replay (τ 0.2 s, k 2.5; low-confidence frames drive the replay, D23/D24)

| Variant | GT-box containment | Fully inside | Centering (mean / p95) | Acceleration (mean / p95) | Crop area | Ski ends inside (Ski 2DPose) |
| --- | --- | --- | --- | --- | --- | --- |
| smoothed | 0.943 [0.911, 0.979] | 0.892 | 0.131 / 0.294 | 1.82 / 7.71 | 0.46 | 99.4% (834/839) [98.4, 100.0] |
| unsmoothed_baseline | 0.910 [0.867, 0.964] | 0.845 | 0.131 / 0.321 | 4.70 / 18.30 | 0.40 | 98.8% (829/839) [97.3, 100.0] |

## Propagation: reference vs. tracker crop (k_pose 1.1; 261 labels in 31 segments)

| Run | Crop | Body @0.1 | Ski @0.1 | Ski @0.05 |
| --- | --- | --- | --- | --- |
| E1 | reference | 95.1% (6387/6716) | 92.3% (2867/3106) | 83.6% (2597/3106) |
| E1 | refined | 84.0% (5639/6716) | 80.0% (2486/3106) | 71.5% (2222/3106) |
| E1 | tracked | 80.2% (5383/6716) | 73.0% (2267/3106) | 62.7% (1946/3106) |
| E2 | reference | 95.9% (6440/6716) | 93.8% (2913/3106) | 85.2% (2647/3106) |
| E2 | refined | 83.9% (5638/6716) | 81.3% (2526/3106) | 72.7% (2257/3106) |
| E2 | tracked | 81.1% (5449/6716) | 75.4% (2342/3106) | 65.6% (2038/3106) |

Tracker-crop cut-off share: body 9.8%, ski 16.8%.

Refined crop (D26: s_min 0.3, k_refine 1.3): cut-off share body 7.6%, ski 10.8%; kept the tracker crop (fallback): E1-s230 4.6% / E1-s231 5.4% / E2-s230 9.2% / E2-s231 6.9%.

| Δ | Group | Δ (points) | 95% interval |
| --- | --- | --- | --- |
| E1 reference − E1 tracked | body @0.1 | +14.9 | [8.2, 23.2] |
| E1 reference − E1 tracked | ski @0.1 | +19.3 | [13.3, 26.3] |
| E1 reference − E1 tracked | ski @0.05 | +21.0 | [15.6, 27.0] |
| E2 reference − E2 tracked | body @0.1 | +14.8 | [7.6, 23.6] |
| E2 reference − E2 tracked | ski @0.1 | +18.4 | [12.5, 25.2] |
| E2 reference − E2 tracked | ski @0.05 | +19.6 | [14.1, 26.2] |
| E2 tracked − E1 tracked | body @0.1 | +1.0 | [-0.8, 2.6] |
| E2 tracked − E1 tracked | ski @0.1 | +2.4 | [1.3, 3.5] |
| E1 refined − E1 tracked | body @0.1 | +3.8 | [1.8, 6.3] |
| E1 refined − E1 tracked | ski @0.1 | +7.1 | [4.6, 9.8] |
| E1 refined − E1 tracked | ski @0.05 | +8.9 | [5.8, 12.2] |
| E2 refined − E2 tracked | body @0.1 | +2.8 | [1.2, 4.5] |
| E2 refined − E2 tracked | ski @0.1 | +5.9 | [3.6, 8.4] |
| E2 refined − E2 tracked | ski @0.05 | +7.1 | [4.7, 9.5] |
| E2 refined − E1 refined | body @0.1 | -0.0 | [-1.9, 1.4] |
| E2 refined − E1 refined | ski @0.1 | +1.3 | [-0.9, 3.0] |

| Stratum | Value | Run | Labels | Body ref → tracked → refined | Ski ref → tracked → refined |
| --- | --- | --- | --- | --- | --- |
| tracker_iou | 0.1<=iou<0.5 | E1 | 129 | 96.0 → 85.5 → 92.4 | 93.5 → 72.7 → 85.6 |
| tracker_iou | 0.1<=iou<0.5 | E2 | 129 | 96.7 → 87.9 → 93.3 | 94.6 → 76.8 → 88.0 |
| tracker_iou | iou<0.1 | E1 | 33 | 91.1 → 2.7 → 9.7 | 89.3 → 1.2 → 5.8 |
| tracker_iou | iou<0.1 | E2 | 33 | 93.5 → 2.6 → 6.8 | 92.7 → 0.9 → 4.6 |
| tracker_iou | iou>=0.5 | E1 | 99 | 95.2 → 95.9 → 94.9 | 91.7 → 91.3 → 92.4 |
| tracker_iou | iou>=0.5 | E2 | 99 | 95.6 → 95.4 → 94.6 | 93.2 → 92.5 → 93.1 |
| time_since_init | 2-5s | E1 | 120 | 96.6 → 74.7 → 79.4 | 95.1 → 70.9 → 77.2 |
| time_since_init | 2-5s | E2 | 120 | 97.3 → 76.7 → 79.8 | 95.9 → 73.3 → 77.4 |
| time_since_init | <2s | E1 | 104 | 93.3 → 87.8 → 90.6 | 88.0 → 76.1 → 84.2 |
| time_since_init | <2s | E2 | 104 | 94.1 → 87.6 → 88.9 | 90.4 → 78.8 → 85.6 |
| time_since_init | >=5s | E1 | 37 | 95.5 → 75.8 → 79.8 | 96.5 → 70.1 → 76.4 |
| time_since_init | >=5s | E2 | 37 | 96.4 → 77.3 → 83.3 | 97.5 → 71.6 → 80.9 |
| status | low_conf | E1 | 70 | 91.7 → 65.4 → 68.7 | 88.2 → 54.9 → 64.7 |
| status | low_conf | E2 | 70 | 93.7 → 64.3 → 66.7 | 90.2 → 57.9 → 64.5 |
| status | ok | E1 | 191 | 96.3 → 85.5 → 89.5 | 93.8 → 79.6 → 85.7 |
| status | ok | E2 | 191 | 96.7 → 87.2 → 90.2 | 95.1 → 81.8 → 87.5 |

