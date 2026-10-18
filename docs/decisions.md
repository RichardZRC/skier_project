# Decisions

One entry per decision: date, choice, reason, and effect on evaluation. Add new entries at the bottom; never rewrite an old one, supersede it.
Seeded from the [six-week plan](../skier_implementation_plan.md) §2.

## D1 — Learned component (2026-10-17)
- Choice: option A. Fine-tune COCO HRNet-W32 to 24 landmarks and run the augmentation study; the tracker stays a fixed pretrained SiamFC.
- Reason: a fixed tracker plus geometric replay has no learned part. Pose fine-tuning is the deep-learning study, and pose is independent of tracking until integration (Step 7).
- Effect on evaluation: the main comparison is E1 vs. E2–E4 ski PCK on the frozen pose split. Tracking is reported as a fixed baseline.

## D2 — Compute (2026-10-17)
- Choice: an AWS GPU instance (g5.xlarge) for all `[GPU]` work. The local GTX 1070 Ti (8 GB) is only a stopgap while the AWS quota is pending.
- Reason: Ruichao has an AWS budget (PROJECT_NOTES, 2026-10-17).
- Effect on evaluation: none directly. Record the instance type and GPU with every run.

## D3 — MMPose usage (2026-10-17)
- Choice: model only. MMPose builds HRNet-W32 and loads the COCO weights; our own plain-PyTorch code does data loading, cropping, heatmap targets, masked loss, the training loop, decoding and metrics.
- Reason: avoids MMPose defaults that silently change results (double 1.25 box padding, visibility parsing, loss reduction, half-body augmentation, the COCO evaluator, automatic LR scaling, flip test, detection files).
- Effect on evaluation: our pipeline must match MMPose's inference on 10 pilot images (mean distance below about 2 px) before E0 or any training (Step 3).

## D4 — Pose split (2026-10-17)
- Choice:
  - whole source videos stay in one partition, about 70/15/15 by image count;
  - at least 2 videos each in val and test;
  - race types (SL/GS/SG/DH) spread across partitions where possible;
  - follow-cam placement decided before any results;
  - the split is frozen before any model comparison.
- Reason: leakage risk comes from the same video, course and athlete, not from near-duplicate frames.
- Effect on evaluation: all pose numbers use `data/manifests/pose_split_v1.json`. Pilot videos are forced into training.

## D5 — Data location (2026-10-17)
- Choice: data, weights and the Python environment live outside any cloud-sync folder. Code reads `SKIER_DATA` and `SKIER_RESULTS` (`src/skier/paths.py`).
- Reason: syncing about 1 GB of data plus a git folder risks corruption.
- Effect on evaluation: none.

## D6 — Model input/output (v1)
- Choice: 256 high × 192 wide input; 24 heatmaps of 64 × 48 (stride 4).
- Reason: matches the COCO HRNet-W32 256×192 checkpoint.
- Effect on evaluation: heatmap coordinate = input coordinate / 4, with no half-pixel shift.

## D7 — Reported groups (v1)
- Choice: body (14) and ski (8) are scored separately; poles (2) are supplementary only.
- Reason: ski accuracy is the research question; poles belong to neither the body nor the ski group.
- Effect on evaluation: separate body and ski PCK columns in every table.

## D8 — Tracking split (v1)
- Choice: SkiTB alpine (AL), the official athlete-disjoint train/val/test split, single-camera (SC) clips.
- Reason: official and athlete-disjoint. Single-camera clips match the core scope, which excludes recovery across camera cuts.
- Effect on evaluation: tracking metrics on AL test SC clips only, computed once in Step 8.

## D9 — Manual annotation (v1)
- Choice: none for the core.
- Reason: the core experiments use the published labels (proposal §3); annotation time only comes out of the 8 h reserve.
- Effect on evaluation: the amateur-footage stretch is qualitative unless the reserve allows 30–60 annotated frames.

## D11 — Follow-cam placement (2026-10-17)
- Choice: option (a). The 6 follow-cam videos (134 images, one segment each) are a separate `followcam` evaluation set: never trained on, outside the 70/15/15 split, reported as their own row. Implemented with `make_pose_split.py --set-aside followcam=auto`.
- Reason: they are the closest stand-in for amateur footage. Forcing all six into test would make them most of the test set's videos, and leaving them to the generator would put them all in training (checked with dry runs of the split search, 2026-10-17).
- Effect on evaluation: 7% fewer training images. The main split stays about 65/16/19. Follow-cam is evaluated once, like test.

## D10 — Pilot videos (2026-10-17)
- Choice: `5UHRvqx1iuQ` (Slalom, 300 images), `oKQFABiOTw8` (Giant Slalom, 385), `zW1bF2PsB0M` (Super G, 114), in `data/manifests/pilot_videos.txt`.
- Reason: they are forced into training, so they are the only videos studied closely before the freeze (overlay review, code sanity checks, the MMPose cross-check). Picking the **larger** video of each two-video race type gives the best achievable split (about 65/16/19 with follow-cam set aside); the smaller ones would push val to 27%.
- Effect on evaluation: none on the metrics. If the overlay review shows a pilot's skier is too small to judge, swap it before the freeze and record the swap here.

## D12 — Heatmap convention and the MMPose cross-check (2026-10-17)
- Choice:
  - Training targets are Gaussians centered **exactly** at `input / 4` (sub-pixel, sigma 2).
  - Decoding copies MMPose 1.x's MSRA decoder verbatim: argmax, then a quarter-pixel step toward the larger neighbor using MMPose's own edge conditions (no step in the outer heatmap pixels), then times 4.
  - The cross-check runs MMPose's `inference_topdown` with `flip_test` **disabled**.
- Reason:
  - MMPose's default MSRA target rounds the center to a whole heatmap pixel, losing up to 2 input px. The sub-pixel target is more precise and only affects training.
  - The decoder must match the COCO checkpoint for E0 and for the cross-check.
  - The COCO config enables flip-test averaging by default, but our pipeline never flips (plan §10, 5.2), so comparing against flip-tested output would measure the wrong thing.
- Effect on evaluation: decode error is at most about 1.4 input px inside the heatmap and about 2.2 px in the outer pixels (tests check both). The cross-check passes only if the mean difference is ≤ 2 original px.

## D13 — No local installs; torch-dependent tests run on AWS (2026-10-17)
- Choice: nothing more is installed on the Mac (no torch, no mmpose). Code that needs torch keeps its imports lazy. Its tests are marked `torch` (and mmpose ones `stack`); they skip locally and run on AWS. Logic is written to be testable with numpy/OpenCV where possible (for example, a numpy reference loss `masked_mse_np`).
- Reason: Ruichao's instruction (2026-10-17).
- Effect on evaluation: none on the metrics. **The first action on AWS, before any training, is a full `pytest`** that includes the `torch` and `stack` tests, so the training loop is verified before GPU time counts.

## D14 — Deadline moved up to Fri 2026-11-20 (2026-10-18)
- Choice: finish everything (final test run, report, demo) by **Friday 2026-11-20**, instead of the original six-week plan ending 2026-11-29. Week-by-week schedule: plan §4. Stretch goals (amateur footage, 3D, coaching) happen only if Week 3 ends on track.
- Reason: Ruichao's target (2026-10-18). It's feasible because the offline code for Steps 1–5 is already written: about 62–70 h of estimated remaining work against about 92 h of capacity.
- Effect on evaluation: none on the protocol. The split freeze, val-only tuning and the test-once rule are unchanged. If the schedule slips, cut in this order: stretch goals → extra seeds → E2/E3 (keep E1 vs. E4). See plan §15.
