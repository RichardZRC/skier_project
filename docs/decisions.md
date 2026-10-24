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

## D15 — SkiTB data, split file, and option A confirmed (2026-10-18)
- Choice:
  - SkiTB arrived; it lives at `$SKIER_DATA/skitb/`. Only `AL/` is used, and `FS/`/`JP/` are never copied to AWS.
  - Tracking uses **`AL/AL_train_val_test_athlete_60-40.json`** (53 / 6 / 41 sequences; 554 / 75 / 471 single-camera clips), the official split named in D8, and every single-camera clip of each sequence.
  - **D1 option A is confirmed.** `PROJECT_NOTES` had deferred A vs. B "until SkiTB arrives", and it has now arrived: pose fine-tuning stays the learning study, and the tracker stays a fixed pretrained SiamFC.
- Reason:
  - The three-way file gives a val partition for tuning the replay, and its test set is identical to the two-way athlete file's test set.
  - For A: the Nov 20 deadline (D14). B would need tracker fine-tuning time and a proposal change approved by the TA.
- Effect on evaluation:
  - The replay τ/k grid is tuned on the 6 val sequences only. Test is run once, in Step 8.
  - Two facts from the audit (`docs/data_audit_skitb.md`):
    - **The athlete split isn't fully athlete-disjoint by the metadata:** 5 athletes in train and test, 10 test sequences. This causes no leakage, since nothing is trained on SkiTB. Report it as "the official athlete split".
    - **The frame rate varies by sequence:** about 25 or about 30 fps, estimated. It must be known per clip before tracking failure episodes or replay timing are scored.

## D16 — Step 6 augmentations, the crop-error set, and the recipe-selection rule (2026-10-18)
- Choice, fixed before any E2–E4 results exist:
  - **Experiments:** E2 = E1 + crop jitter; E3 = E1 + motion blur + occlusion; E4 = E1 + all three. Everything else matches E1 exactly (`configs/pose/e2–e4.yaml`; a test checks this).
  - **Crop jitter:** probability 1. The reference-box center moves by U(−0.10, 0.10) of the box's width and height, independently per axis, and its size by U(0.9, 1.1). The labels follow through the single crop transform.
  - **Motion blur:** probability 0.5. A line kernel at a random direction, with length 3, 5, 7 or 9 px at model resolution. Odd lengths only, so the kernel never shifts the image against the labels.
  - **Occlusion:** probability 0.5. One solid random-color rectangle covering 5–15% of the crop, with width/height ratio log-uniform in 0.5–2, fully inside the crop.
  - **Occlusion label policy:** points under the patch **stay supervised**, since their true position is known. Unreliable labels stay masked as always. Neither pixel augmentation changes targets or weights.
  - **Paired design:** one fixed-length vector of uniforms per `(seed, epoch, index)`, one slot per parameter.
    - E1–E4 see identical flips, rotations, scales and data order.
    - E2 and E4 share jitter values; E3 and E4 share blur and occlusion values.
  - **Crop-error set** (`data/manifests/crop_error_set_v1.json`, val):
    - Per image, three levels (mild 0.05, moderate 0.10, severe 0.20). At each level the reference box's center moves by exactly that radius, in box units, in a random direction, and its size by a factor in [1/(1+r), 1+r].
    - Seed 230; each sample's draws depend only on (seed, sample ID).
    - Only perturbation parameters are stored, never coordinates. The file is frozen like the split.
    - Reliable points the crop cuts off count as misses, and their share is reported.
  - **Strata in Step 6:** discipline and view flag on val only. **Follow-cam is not used in Step 6.** D11 makes it a once-only set like test (plan §11's stratum list is superseded).
  - **Selection rule:**
    1. Drop candidates whose clean val body PCK@0.1 is more than 2 points below E1.
    2. Take the highest clean val ski PCK@0.1. If the top two are within 1 point, take the one with the higher moderate crop-error ski PCK@0.1.
    3. If none beats E1, keep E1 and report the negative result.

    `compare_experiments.py` prints the rule's pick. Ruichao makes the final choice and records it here.
- Reason:
  - The values implement plan §11's ranges.
  - Probability 0.5 for blur and occlusion keeps half of the samples clean, so clean accuracy isn't traded away by default.
  - The severe level lies outside E2's training jitter (at most 0.14 radially), so the set also measures extrapolation.
  - The levels are fixed before any tracker results, so they can't be fitted to them.
  - Writing the rule down first prevents choosing the recipe after seeing the numbers.
- Effect on evaluation:
  - The Step 6 table reports, per run, clean body / body-shared / ski PCK, ski PCK at each crop-error level, the cut-off share, and Δ vs. E1, with counts.
  - Bootstrap intervals come in Step 8.

## D17 — GPU work moves to the Windows laptop for now; AWS stays for later (2026-10-24)
- Choice:
  - Ruichao's Windows laptop (RTX 4070 Laptop GPU, probably 8 GB) becomes the GPU machine, running WSL2 Ubuntu 22.04 with the same pinned stack. Setup: `docs/laptop_setup.md`.
  - It takes everything doable now: the full test gate, smoke tests, MMPose cross-check, E0, E1 (overfit, profile, full), E2–E4, evaluation and comparison, and the tracker and replay runs.
  - **Nothing is dropped.** AWS (g5.xlarge, guide §1C) stays in the plan for heavier or parallel work, such as extra seeds, once the new account's verification completes.
  - The Mac keeps coding, reviews and the split freeze, with no installs (D13).
- Reason:
  - AWS credits are redeemed, but the account's verification is still in progress. Its GPU-quota request was rejected automatically, and the Standard-instance quota is 1 vCPU.
  - The laptop removes that wait. The dataset is small (about 1,500 training images), so a laptop GPU is enough for the core runs.
  - This supersedes D2's "AWS for all GPU work"; the original 1070 Ti stopgap is no longer needed.
- Effect on evaluation:
  - **Runs compared in one table come from the same kind of machine with the same batch size.** If seeds move to AWS later, E1 is retrained there too.
  - If 8 GB can't fit batch 16, the batch size changes before any full run, applies to E1–E4 alike, and is recorded as a new decision.
  - Each run records its GPU (`check_env.py`'s `env.json`).
