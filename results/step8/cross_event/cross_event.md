# Cross-event breakdown (post hoc; frozen predictions, no new evaluation)

Training event(s): Almaty, Kazakhstan. Event = the info JSON's video location.
PCK (%) seed-pooled, 95% interval (D25: segments within videos; follow-cam: videos).

## Pose model, reference crops

**test, same event as training** (uLW74013Wp0; 284 samples)

| Run | body@0.1 | ski@0.1 | ski@0.05 |
| --- | --- | --- | --- |
| E1 | 98.9 [98.3, 99.4] (7411/7490) | 96.7 [95.7, 97.6] (3413/3528) | 93.4 [91.5, 95.0] (3295/3528) |
| E2 | 99.1 [98.4, 99.5] (7419/7490) | 98.1 [97.0, 99.0] (3462/3528) | 95.6 [93.8, 97.0] (3372/3528) |
| Δ E2 − E1 body@0.1 | +0.1 [-0.0, +0.2] | | |
| Δ E2 − E1 ski@0.1 | +1.4 [+0.8, +2.1] | | |
| Δ E2 − E1 ski@0.05 | +2.2 [+1.4, +3.0] | | |

**test, other events** (NrWcP1s3QC0, dpFlXMvvZ9E; 76 samples)

| Run | body@0.1 | ski@0.1 | ski@0.05 |
| --- | --- | --- | --- |
| E1 | 85.7 [77.5, 91.8] (1575/1838) | 83.5 [77.0, 88.3] (723/866) | 75.4 [67.7, 81.4] (653/866) |
| E2 | 82.8 [75.5, 88.9] (1522/1838) | 81.2 [72.9, 87.6] (703/866) | 72.4 [64.3, 79.0] (627/866) |
| Δ E2 − E1 body@0.1 | -2.9 [-6.5, +0.6] | | |
| Δ E2 − E1 ski@0.1 | -2.3 [-6.4, +0.9] | | |
| Δ E2 − E1 ski@0.05 | -3.0 [-6.5, +0.5] | | |

**test, all** (NrWcP1s3QC0, dpFlXMvvZ9E, uLW74013Wp0; 360 samples)

| Run | body@0.1 | ski@0.1 | ski@0.05 |
| --- | --- | --- | --- |
| E1 | 96.3 [94.6, 97.6] (8986/9328) | 94.1 [92.6, 95.3] (4136/4394) | 89.8 [87.6, 91.7] (3948/4394) |
| E2 | 95.9 [94.1, 97.2] (8941/9328) | 94.8 [93.0, 96.3] (4165/4394) | 91.0 [88.7, 92.9] (3999/4394) |
| Δ E2 − E1 body@0.1 | -0.5 [-1.3, +0.2] | | |
| Δ E2 − E1 ski@0.1 | +0.7 [-0.3, +1.5] | | |
| Δ E2 − E1 ski@0.05 | +1.2 [+0.2, +2.2] | | |

**val (all other events)** (SVYdH614G_M, qxfgw1Kd98A, v7PADD2_I00; 293 samples)

| Run | body@0.1 | ski@0.1 | ski@0.05 |
| --- | --- | --- | --- |
| E1 | 95.2 [93.2, 96.6] (7155/7518) | 91.9 [88.8, 94.1] (3220/3504) | 82.9 [78.0, 86.6] (2904/3504) |
| E2 | 95.5 [93.0, 97.1] (7178/7518) | 93.1 [89.8, 95.4] (3263/3504) | 84.3 [79.5, 88.1] (2955/3504) |
| Δ E2 − E1 body@0.1 | +0.3 [-0.7, +1.2] | | |
| Δ E2 − E1 ski@0.1 | +1.2 [+0.1, +2.3] | | |
| Δ E2 − E1 ski@0.05 | +1.5 [+0.4, +2.6] | | |

**follow-cam (other camera setup)** (37HtpPpdvp0, 3wkVqupcMAk, FW1S3VAKui8, LjXeTB3Wl10, OeDt6D_9Rh4, nq2qj0fhYjc; 134 samples)

| Run | body@0.1 | ski@0.1 | ski@0.05 |
| --- | --- | --- | --- |
| E1 | 97.4 [95.2, 98.5] (3359/3450) | 73.5 [69.0, 76.3] (1376/1872) | 59.0 [56.6, 60.6] (1104/1872) |
| E2 | 97.5 [95.4, 98.8] (3364/3450) | 78.1 [73.6, 81.4] (1462/1872) | 61.0 [57.9, 64.4] (1142/1872) |
| Δ E2 − E1 body@0.1 | +0.1 [-0.2, +0.8] | | |
| Δ E2 − E1 ski@0.1 | +4.6 [+2.9, +6.7] | | |
| Δ E2 − E1 ski@0.05 | +2.0 [+0.1, +4.8] | | |

## System, SiamFC tracks (main result), test

**test, same event as training** (uLW74013Wp0; 262 samples)

| Run | body@0.1 | ski@0.1 |
| --- | --- | --- |
| E2 reference | 99.1 [98.6, 99.6] (6845/6904) | 98.3 [97.3, 99.1] (3192/3246) |
| E2 tracked | 78.0 [66.1, 88.5] (5387/6904) | 65.6 [53.6, 76.0] (2129/3246) |
| E2 refined | 82.1 [70.2, 92.0] (5670/6904) | 75.7 [62.9, 86.1] (2458/3246) |
| Δ E2 reference − E2 tracked body@0.1 | +21.1 [+10.6, +32.9] | | |
| Δ E2 reference − E2 tracked ski@0.1 | +32.7 [+22.4, +44.6] | | |
| Δ E2 refined − E2 tracked body@0.1 | +4.1 [+1.6, +6.9] | | |
| Δ E2 refined − E2 tracked ski@0.1 | +10.1 [+7.3, +13.4] | | |

**test, other events** (NrWcP1s3QC0, dpFlXMvvZ9E; 55 samples)

| Run | body@0.1 | ski@0.1 |
| --- | --- | --- |
| E2 reference | 86.1 [80.0, 91.0] (1124/1306) | 82.2 [73.8, 88.6] (526/640) |
| E2 tracked | 83.6 [78.3, 88.3] (1092/1306) | 78.4 [70.5, 84.9] (502/640) |
| E2 refined | 83.8 [77.3, 89.7] (1095/1306) | 83.4 [75.4, 89.6] (534/640) |
| Δ E2 reference − E2 tracked body@0.1 | +2.5 [-1.4, +5.5] | | |
| Δ E2 reference − E2 tracked ski@0.1 | +3.7 [-2.7, +9.1] | | |
| Δ E2 refined − E2 tracked body@0.1 | +0.2 [-3.2, +3.4] | | |
| Δ E2 refined − E2 tracked ski@0.1 | +5.0 [-0.6, +10.9] | | |

**test, all** (NrWcP1s3QC0, dpFlXMvvZ9E, uLW74013Wp0; 317 samples)

| Run | body@0.1 | ski@0.1 |
| --- | --- | --- |
| E2 reference | 97.1 [95.8, 98.0] (7969/8210) | 95.7 [94.1, 97.0] (3718/3886) |
| E2 tracked | 78.9 [69.0, 87.7] (6479/8210) | 67.7 [57.7, 76.3] (2631/3886) |
| E2 refined | 82.4 [72.7, 90.6] (6765/8210) | 77.0 [66.5, 85.7] (2992/3886) |
| Δ E2 reference − E2 tracked body@0.1 | +18.1 [+9.3, +27.9] | | |
| Δ E2 reference − E2 tracked ski@0.1 | +28.0 [+19.3, +37.7] | | |
| Δ E2 refined − E2 tracked body@0.1 | +3.5 [+1.3, +6.0] | | |
| Δ E2 refined − E2 tracked ski@0.1 | +9.3 [+6.7, +12.1] | | |

## System, STARK tracks (D28 extra), test

**test, same event as training** (uLW74013Wp0; 262 samples)

| Run | body@0.1 | ski@0.1 |
| --- | --- | --- |
| E2 reference | 99.1 [98.6, 99.6] (6845/6904) | 98.3 [97.3, 99.1] (3192/3246) |
| E2 tracked | 98.6 [97.7, 99.4] (6810/6904) | 87.2 [79.7, 92.2] (2832/3246) |
| E2 refined | 98.8 [98.1, 99.4] (6823/6904) | 94.2 [90.2, 96.8] (3057/3246) |
| Δ E2 reference − E2 tracked body@0.1 | +0.5 [-0.2, +1.4] | | |
| Δ E2 reference − E2 tracked ski@0.1 | +11.1 [+6.2, +18.3] | | |
| Δ E2 refined − E2 tracked body@0.1 | +0.2 [-0.6, +1.1] | | |
| Δ E2 refined − E2 tracked ski@0.1 | +6.9 [+3.7, +11.0] | | |

**test, other events** (NrWcP1s3QC0, dpFlXMvvZ9E; 55 samples)

| Run | body@0.1 | ski@0.1 |
| --- | --- | --- |
| E2 reference | 86.1 [80.0, 91.0] (1124/1306) | 82.2 [73.8, 88.6] (526/640) |
| E2 tracked | 85.3 [79.5, 90.4] (1114/1306) | 82.8 [74.9, 89.1] (530/640) |
| E2 refined | 83.5 [77.1, 89.4] (1090/1306) | 81.7 [73.4, 88.5] (523/640) |
| Δ E2 reference − E2 tracked body@0.1 | +0.8 [-2.8, +3.9] | | |
| Δ E2 reference − E2 tracked ski@0.1 | -0.6 [-5.6, +3.6] | | |
| Δ E2 refined − E2 tracked body@0.1 | -1.8 [-4.2, +0.4] | | |
| Δ E2 refined − E2 tracked ski@0.1 | -1.1 [-5.0, +2.4] | | |

**test, all** (NrWcP1s3QC0, dpFlXMvvZ9E, uLW74013Wp0; 317 samples)

| Run | body@0.1 | ski@0.1 |
| --- | --- | --- |
| E2 reference | 97.1 [95.8, 98.0] (7969/8210) | 95.7 [94.1, 97.0] (3718/3886) |
| E2 tracked | 96.5 [95.1, 97.6] (7924/8210) | 86.5 [80.0, 90.9] (3362/3886) |
| E2 refined | 96.4 [94.8, 97.6] (7913/8210) | 92.1 [88.4, 94.7] (3580/3886) |
| Δ E2 reference − E2 tracked body@0.1 | +0.5 [-0.3, +1.5] | | |
| Δ E2 reference − E2 tracked ski@0.1 | +9.2 [+4.9, +15.2] | | |
| Δ E2 refined − E2 tracked body@0.1 | -0.1 [-0.9, +0.7] | | |
| Δ E2 refined − E2 tracked ski@0.1 | +5.6 [+2.7, +9.0] | | |

