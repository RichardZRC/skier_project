# PL (pseudo-label student, D29) vs. E2: followcam (post-freeze extra)

PCK (%) seed-pooled, paired 95% intervals (D25: segments within videos; follow-cam: videos).

## Pose model, reference crops (clean)

**follow-cam (other camera setup)** (37HtpPpdvp0, 3wkVqupcMAk, FW1S3VAKui8, LjXeTB3Wl10, OeDt6D_9Rh4, nq2qj0fhYjc; 134 samples)

| Run | body@0.1 | ski@0.1 | ski@0.05 |
| --- | --- | --- | --- |
| E2 | 97.5 [95.4, 98.8] (3364/3450) | 78.1 [73.6, 81.4] (1462/1872) | 61.0 [57.9, 64.4] (1142/1872) |
| PL | 98.1 [97.0, 98.8] (3386/3450) | 77.2 [74.5, 79.5] (1446/1872) | 60.2 [58.2, 61.8] (1127/1872) |
| Δ PL − E2 body@0.1 | +0.6 [-0.1, +1.8] | | |
| Δ PL − E2 ski@0.1 | -0.9 [-3.8, +1.1] | | |
| Δ PL − E2 ski@0.05 | -0.8 [-2.8, +0.7] | | |

