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
