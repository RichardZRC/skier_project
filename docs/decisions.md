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

## D18 — Two pins added to the GPU stack: mmdet 3.2.0, opencv-python < 4.12 (2026-10-24)
- Choice:
  - Install **`mmdet==3.2.0`** after mmpose.
  - Re-pin **`opencv-python<4.12`** together with `numpy<2` (the laptop got 4.11.0.86).
  - Both are now in `requirements-pose.txt`, the laptop guide (2.3) and guide 1C.7.
- Reason:
  - **mmdet:** mmpose 1.3.2 imports its RTMO head whenever `mmpose.models` loads, and that head imports mmdet. mmpose lists mmdet only as an optional extra (`>=3.0.0,<3.3.0`). Without it, every `mmpose.apis` import fails, and so do `build_hrnet`, the smoke test, the cross-check, E0 and training. On the laptop, 2 HRNet tests failed in the gate. mmdet 3.2.0 supports mmcv 2.1.0, is pure Python, and adds only pycocotools and terminaltables; no pinned package changed (dry run).
  - **OpenCV:** mmcv asks for `opencv-python>=3`, so pip installed 5.0.0.93, which declares `numpy>=2` and conflicts with our `numpy<2` pin (`pip check` failed). 4.11 is the newest line that accepts numpy 1.x.
  - Ruichao approved the mmdet install (2026-10-24).
- Effect on evaluation: none. mmdet's code is imported but never runs in our pipeline. After both fixes the laptop gate passed (257 tests, `check_env --strict` exit 0).

## D19 — Pose split and evaluation protocol frozen (2026-10-24)
- Choice:
  - `data/manifests/pose_split_v1.json` is frozen. Content SHA-256 `46bd127b0e51c1293859c7ab86dab1f0759511fbbdfd9dad270a632443bb2e98`, written by `make_pose_split.py --set-aside followcam=auto` (seed 230, targets 70/15/15).
  - The partitions:

    | Partition | Videos | Images |
    | --- | --- | ---: |
    | train | `5UHRvqx1iuQ`, `he3w2n9WvrI`, `oKQFABiOTw8`, `zW1bF2PsB0M` | 1,194 (65%) |
    | val | `SVYdH614G_M`, `qxfgw1Kd98A`, `v7PADD2_I00` | 293 (16%) |
    | test | `NrWcP1s3QC0`, `dpFlXMvvZ9E`, `uLW74013Wp0` | 360 (19%) |
    | followcam | the 6 follow-cam videos (D11) | 134 |

    1 record is excluded (`oKQFABiOTw8/27/oKQFABiOTw8_27_00020`, train).
  - `docs/evaluation_protocol.md` is frozen with it.
- Reason:
  - The 1F overlay review passed: 30 pilot overlays, all six checks, no problems (`docs/data_audit_ski2dpose.md` §1F).
  - The split equals the 2026-10-17 dry run, and the frozen file matches today's dry run.
  - It meets D4 and D11: 3 videos each in val and test; at least 400 reliable ski points in each (val 1,752, test 2,197); pilots in train; follow-cam set aside.
  - Ruichao approved it after seeing the summary.
- Effect on evaluation:
  - Every pose number from now on uses this manifest. Checkpoints and metrics record its SHA-256, and `eval_pose.py` refuses a checkpoint trained on another split.
  - The 1G alignment check isn't needed for the freeze; only Step 7 waits for it.

## D20 — A convergence trial before any recipe choice (2026-10-25)
- Choice: run two diagnostic trainings before choosing a recipe or rerunning E1–E4. Written down before either has run.
  - **E1-seed231:** `configs/pose/e1.yaml` with `--seed 231`, everything else unchanged. It asks whether E1 learns the ski tails reliably or got lucky.
  - **E3-ft150:** `configs/pose/e3.yaml` with `--epochs 150` (150 fine-tune epochs instead of 50; the ×0.1 steps move with it to epochs 90 and 127). It asks whether longer training lets an augmented run learn the tails.
  - **Measured:** per-joint PCK@0.1 of each `best.pt` on the training images (no augmentation) and on clean val, with the ski tails as the main readout. `results/step6/train_fit.py` gives the E1–E4 reference values.
  - **Reading, fixed in advance.** A run "learns the tails" if its training-image tail PCK@0.1 is at least 60% on both sides (E1: 73 / 78%; E2–E4: about 20%).

    | E1-seed231 | E3-ft150 | Then |
    | --- | --- | --- |
    | learns | learns | Rerun E1–E4 with the longer schedule (a new decision). |
    | fails | learns | Same, and treat seed variance as large: 2+ seeds per recipe. |
    | learns or fails | fails | Longer training isn't enough; investigate further before any rerun. |

  - **Neither run joins the Step 6 table.** They are diagnostics, kept in `results/step6/`. The D16 rule and settings are unchanged.
- Reason:
  - The Step 6 comparison is confounded by convergence (`results/step6/diagnostics.md`). E2–E4 miss the ski tails even on their own training images, and E3's training targets are bit-identical to E1's.
  - A cheap trial first (working rule): 2 runs, about 1 h, before a 2.5 h rerun of all four.
  - Ruichao chose this option (A) on 2026-10-25.
- Effect on evaluation: none yet. Val is used only for the diagnostic readout, and test is untouched. Any schedule change it leads to will apply to E1–E4 alike, through its own decision.

## D21 — Step 6 rerun: 150 fine-tune epochs, two seeds per recipe, selection on the seed means (2026-10-26)
- Choice, fixed before any rerun result exists:
  - **Schedule:** `finetune_epochs: 150` in all of `configs/pose/e1.yaml`–`e4.yaml`. The ×0.1 steps move with it, to fine-tune epochs 90 and 127. Nothing else changes, and E2–E4 still differ from E1 only in augmentation (the test still checks this). This supersedes the 50-epoch schedule for every Step 6 number.
  - **Seeds:** 230 and 231 for every recipe; the paired design (D16) holds within each seed. Runs go to `$SKIER_RESULTS/pose/d21/E{n}-s{seed}/`, seed 230 first.
  - **Reuse:** the D20 run E3-ft150 is exactly E3 seed 230 under this schedule (same config, seed, split and code), so it serves as `E3-s230`. That leaves 7 new runs.
  - **Selection:** D16's rule, unchanged, applied to the **mean over the two seeds** of each metric. All seeds see the same val samples, so the mean equals pooling the hits and totals of both seeds. Each seed is also reported on its own.
  - **The 50-epoch E1–E4 runs and the D20 runs** stay in `results/step5/` and `results/step6/` as the record of the convergence problem. They don't enter the final table.
- Reason:
  - D20's pre-registered outcome: E1-seed231 fails on the ski tails and E3-ft150 learns them. So the 50-epoch schedule is unreliable for every recipe, and seed variance is large (`results/step6/diagnostics.md` §5).
  - In E3-ft150 the tails were learned between epochs 60 and 80, before its first LR step at epoch 93. With 50 epochs, the first step comes at epoch 33.
  - Ruichao approved this on 2026-10-26.
- Effect on evaluation:
  - The Step 6 table reports the seed means, with per-seed values next to them. The D16 rule and the crop-error set are unchanged.
  - Same machine (the RTX 4070 laptop) and batch size 16 for every run (D17).
  - Cost: 7 runs × about 38 min ≈ 4.5 h.
  - D14's cut order changes: a second seed is now needed for a trustworthy comparison, so if time is short, E2 and E3 are cut before the second seed.

## D22 — SkiTB frame rate per sequence: the estimate snapped to 25 or 30 FPS (2026-10-26)
- Choice:
  - Each AL sequence gets the frame rate in **`data/manifests/skitb_fps_v1.csv`**: the nearer of 25 and 30 to (last − first `MC/` frame index) ÷ official `AthleteTime` (`results/step3/skitb_fps_estimate.py`). The result: 18 sequences at 25 and 82 at 30. Val has AL0027 at 25 and the other 5 at 30; test has 8 at 25 and 33 at 30.
  - `run_tracker.py` (`--skitb-root`) and `make_replay.py` (`--grid`, or a single SkiTB clip via `--gt-clip-dir`) use this table by default. `--fps` stays as an explicit override for one rate for all clips. A sequence missing from the table raises, and a value other than 25 or 30 is refused.
  - The manifest is frozen like the split. Changing it needs a new decision.
- Reason:
  - The files don't store the frame rate, and the toolkit doesn't either (its "fps" is tracker speed).
  - The SkiTB paper states "The videos have a framerate of 25/30 FPS" (arXiv 2312.09723, §4), but no per-video rate.
  - The estimates fall in two well-separated groups, 23.4–25.8 and 29.1–31.5. Nothing lies between 25.83 and 29.08, so snapping is unambiguous. AL0098 (37.9, a race-time/span mismatch) goes to 30, the nearer allowed value.
  - The alternative, 30 for every clip as the toolkit's own time metrics assume, would make durations in the 18 slower sequences 20% too short.
  - Ruichao approved this on 2026-10-26.
- Effect on evaluation:
  - Only time-based numbers change: failure-episode durations (≥ 0.5 s), and all replay timing (τ in seconds, acceleration per s²). IoU, the success curve and AUC count frames and don't depend on it.
  - The report states the rates as estimates ("25 or 30 FPS per sequence, from frame span ÷ race time") and names the manifest.

## D23 — Pose recipe E2 (crop jitter); replay defaults τ 0.2 s, k 2.5 (2026-10-27)
- Choice:
  - **Pose recipe: E2** (`configs/pose/e2.yaml`: E1 + crop jitter, 150 fine-tune epochs). Both seed checkpoints (`$SKIER_RESULTS/pose/d21/E2-s230/`, `E2-s231/`) are kept. The Step 7 and Step 8 plans say which checkpoint each later step uses.
  - **Replay defaults:** τ 0.2 s (`tau_pos` 0.2; `tau_grow` 0.1 and `tau_shrink` 0.6 keep the grid's 1/2 and 3× ratios; `tau_urgent` unchanged) and **k 2.5** (was 2.0), in `configs/replay/default.yaml` and the `ReplayConfig` defaults. `lost_score` stays off; Step 7 sets the tracker-confidence handling.
- Reason:
  - **Recipe:** D16's pre-registered rule, applied to the seed means as D21 requires, picks E2 (`results/step6/d21/compare_val_seedmean/`). Val PCK@0.1 seed means, E2 vs. E1: ski 93.1 vs. 91.9, body 95.5 vs. 95.2, moderate crop-error ski 92.9 vs. 90.4. Each seed alone also picks E2. All 8 runs learn the ski tails on training images (`results/step6/d21/train_fit_d21.txt`), so the comparison isn't confounded by convergence.
  - **Replay** (SkiTB val grid, 75 clips, `results/step4/grid_val.csv`), τ 0.2 with k 2.5 vs. k 2.0:
    - box containment 0.857 vs. 0.841, fully inside 78% vs. 75%;
    - mean acceleration 1.21 vs. 1.65 (p95 3.56 vs. 6.47), centering error 0.179 vs. 0.190;
    - cost: the crop covers 34% of the frame instead of 28%, so the skier looks smaller.
    - Unsmoothed at k 2.5, mean acceleration is 4.72. τ 0.4 worsened centering.
  - Ruichao accepted both recommendations on 2026-10-27.
- Effect on evaluation:
  - The Step 8 pose table reports E0, E1 and E2. E3 and E4 stay in the val comparison as the augmentation study.
  - Replay numbers from now on use these defaults. The grid stays as the record of the trade-off.
  - Both choices were made on val only. They are frozen with everything else before the single test run (Step 8).

## D24 — Step 7 integration settings: confidence gating on the raw SiamFC score, k_pose 1.1 (2026-10-27)
- Choice, each by a rule written into the approved Step 7 plan before its run. All are in `configs/integration/default.yaml`.
  - **Tracker confidence:**
    - `conf_signal: raw`, `conf_threshold: 3.944619e-05`. A tracked frame whose SiamFC score is below it is `low_conf`.
    - The rule: ROC AUC for predicting failure (IoU < 0.1) on SkiTB val visible frames, excluding the init frame. Take the better of the two pre-registered signals; threshold at Youden's J; gate only if AUC ≥ 0.70.
    - Result: raw AUC 0.83, relative (score ÷ median of the clip's first 10 tracked frames) 0.63. At the threshold, TPR 0.80 and FPR 0.29. 12,925 frames, 3,817 failures (`results/step7/confidence_val.json`).
  - **Replay:** `replay_uses_status: true`. Low-confidence frames drive the replay as lost: hold the centre, zoom out.
    - The rule: only if that raises mean GT-box containment on SkiTB val. It does: 0.943 vs. 0.857, fully inside 89% vs. 78%.
    - The cost, reported alongside: the crop covers 46% of the frame instead of 34%, and mean acceleration is 1.82 instead of 1.21. 44% of val frames are `low_conf`.
    - `ReplayConfig.lost_score` stays off; the status carries the gating.
  - **Pose crop from the tracker box:** `k_pose: 1.1`, i.e. `expand_box(tracker_box, 1.1)` and then the model's 1.25 padding.
    - The rule: the highest mean of tracked body and ski PCK@0.1 for E2 over both seeds; within 0.5 points is a tie, and a tie goes to the smaller value.
    - The grid {1.0, 1.1, 1.2, 1.35, 1.5} gave 77.2 / **78.3** / 77.9 / 77.4 / 76.9.
  - **Checkpoints:** E2-s230 for `process_clip` and the demos (the primary seed, fixed in advance). The propagation table reports E1 and E2, both seeds.
  - **Track CSV scores** are written with 7 significant digits (`.6e`; they were 5 decimals). The SkiTB val rerun reproduced every Step 3 box exactly (pooled IoU 0.363).
- Reason:
  - The approved Step 7 plan (2026-10-27) fixed every rule above before its run. All runs used val only.
  - The val alignment check passed for all 32 segments: all 293 labels are their own clip frame at offset 0, and 271 also pass the margin rule. All clips are 1280×720 at 25 fps.
- Effect on evaluation:
  - **Propagation on val** (261 scored labels in 32 segments; E2 seed means; PCK@0.1, reference → tracked crop):
    - body 95.9 → 81.1, ski 93.8 → 75.4; the tracker crop cuts off 16.8% of the reliable ski labels;
    - E1: body 95.1 → 80.2, ski 92.3 → 73.0.
  - **By tracker IoU vs. the reference box**, E2 ski:
    - IoU ≥ 0.5 (99 labels): 93.2 → 92.5;
    - 0.1–0.5 (129): 94.6 → 76.8;
    - IoU < 0.1 (33): 92.7 → 0.9.
    - So the loss is tracker drift, not a train/inference crop mismatch.
  - **By status**, E2 ski: `low_conf` 57.9 (70 labels), ok 81.8 (191).
  - Step 8 runs these settings unchanged on test. A later tracker change (7b) needs its own decision and a rerun of the confidence check.

## D25 — Bootstrap unit: segments within videos (Ski 2DPose), sequences (SkiTB) (2026-10-28)
- Choice, recorded before any test number exists:
  - **Ski 2DPose** (pose, propagation): resample **segments with replacement, within each source video** (a stratified cluster bootstrap). The videos are held fixed, so an interval is conditional on these videos. Per-video rows always appear next to the pooled number, so the between-video spread stays visible.
  - **SkiTB** (tracking, replay): resample **sequences** with replacement; each sequence brings all its SC clips.
  - **Settings:** 10,000 resamples, seed 230, percentile 95% intervals. The statistic is recomputed from pooled counts (hits and totals, IoU sums and frame counts) in each resample. The toolkit F-score is recomputed as the max of the mean curve.
  - **Seeds:** E1 and E2 are pooled over seeds 230 and 231 inside every resample (D21). The seed spread is reported separately as per-seed values, never mixed into the interval (plan §13).
  - Code: `skier.eval.bootstrap`.
- Reason:
  - The protocol (`docs/evaluation_protocol.md`: "a bootstrap over source videos") assumed more videos. Test has 3, and `uLW74013Wp0` holds 1,764 of the 2,197 reliable ski labels.
  - Three videos give only 10 distinct resamples, so a video-level interval would mostly say which video was drawn.
  - Test has 43 segments, val 32. Segments are separate camera sequences, the closest independent unit below the video.
  - The change is made for the count of units, never because of a score; test is still unseen.
  - Ruichao approved the Step 8 plan containing it on 2026-10-28.
- Effect on evaluation:
  - Every paired difference in the report gets a segment- or sequence-level 95% interval, labelled as such.
  - The video-level spread is shown through the per-video rows instead.
  - The protocol's bootstrap line is superseded by this decision.

## D26 — Pose-guided crop refinement (Step 7c): method, grid, selection rule and adoption gate (2026-11-08)
- Choice, recorded before any refinement number exists:
  - **Method:** pass 1 is the current tracker crop (`expand_box(tracker_box, k_pose)`, then the model's 1.25 padding). The refined box is the tight box of the pass-1 **body + ski** predictions (poles excluded, as in `geometry.reference_box`) that are finite and have a heatmap score ≥ `s_min`, scaled by `expand_box(·, k_refine)`. Pass 2 crops it with the usual 1.25 padding and 3:4 fit and predicts again. One refinement step only.
  - **Fallback:** fewer than 4 qualifying points, or a box diagonal under 1 px, keeps the pass-1 prediction and crop. The fallback share is reported.
  - **No labels:** the refined box is built from predictions only.
  - **Grid, on val:** `s_min` ∈ {0.1, 0.2, 0.3, 0.5}, `k_refine` ∈ {1.0, 1.15, 1.3}. Heatmap targets peak at 1.0, but the scale of the maxima on bad crops is unknown, so it is chosen on val.
  - **Selection rule:** the highest E2 seed-mean of (refined body + ski PCK@0.1) / 2 over all scored val labels. Settings within 0.5 points of the best tie; a tie goes to the `k_refine` closest to 1.0, then to the higher `s_min`.
  - **Adoption gate:** refinement becomes the system default (`pose_refine: true`, used by `process_clip`) only if the chosen setting beats the single tracker crop by more than 0.5 points on that E2 mean.
  - **Test:** the single Step 8 run reports reference, tracker and refined crops at the chosen setting whatever the gate says, with refined − tracked intervals (D25).
  - **Unchanged:** `k_pose` 1.1, the checkpoints, the tracker, the replay, the bootstrap.
- Reason:
  - The biggest measured error is the tracker crop (D24): E2 loses 18.4 ski points on val. On the 129 labels whose tracker box only partly overlaps the reference box (IoU 0.1–0.5), ski PCK drops 94.6 → 76.8 while body PCK stays at 87.9, so the model still finds the skier in a badly framed crop.
  - It adds a method beyond fine-tuning, with the single tracker crop as baseline and the reference crop as upper bound (CS230 handout; Ruichao, 2026-11-08, after the milestone review).
- Effect on evaluation:
  - The propagation tables and the strata figure get a third crop. The test freeze becomes D27.
- Result on val (2026-11-08; the selection run, `$SKIER_RESULTS/integration/refine-val`, summary in `results/step7/refine_val_summary.json`):
  - **Pick:** `s_min` 0.3, `k_refine` 1.3. The E2 seed-mean of body and ski is 82.6 vs. 78.3 for the single tracker crop (gain +4.4 > 0.5), so **`pose_refine: true`** (`configs/integration/default.yaml`).
    - The best setting was `s_min` 0.2, `k_refine` 1.3 (83.1); 0.3 is within 0.5 points and wins the tie on the higher `s_min`.
    - Every `s_min` row is best at `k_refine` 1.3, the largest value in the grid. The grid is not extended after the fact; the report says the optimum may lie beyond it.
  - **E2, both seeds pooled, PCK@0.1 (reference → tracker crop → refined):**
    - all 261 labels: body 95.9 → 81.1 → 83.9, ski 93.8 → 75.4 → 81.3; ski @0.05 85.2 → 65.6 → 72.7;
    - tracker IoU 0.1–0.5 (129 labels): body 96.7 → 87.9 → 93.3, ski 94.6 → 76.8 → 88.0;
    - IoU ≥ 0.5 (99): body 95.6 → 95.4 → 94.6, ski 93.2 → 92.5 → 93.1;
    - IoU < 0.1 (33): ski 92.7 → 0.9 → 4.6.
  - The ski labels cut off by the crop drop from 16.8% to 10.8% (E2). The fallback share is 5–9% per checkpoint.
  - The reference and tracker-crop numbers equal the Step 7 run exactly.

## D27 — Test freeze: the single Step 8 test run, and two reporting fixes after it (2026-11-12)
- **The run:** `bash results/step8/run_final.sh test` on the laptop, 2026-11-12 from 20:34, at commit `bf546e3`. `freeze.json` (14 files hashed: splits, frame-rate table, configs, the COCO and SiamFC weights, the four checkpoints) was written before any output.
  - Not waiting for TA feedback (Ruichao, 2026-11-12: everything must be finished by Fri 2026-11-20).
  - 7b (STARK) deferred (Ruichao): if it is done later, it is reported as a labeled extra experiment. SiamFC stays the main result, because the test is used once.
  - Before the run, a val-only check of the 4–8% pole PCK found no bug: 48% of the visible pole-basket labels lie outside the body + ski crop (see `PROJECT_NOTES.md`).
- **Every evaluation step finished:**
  - pose on test (E0, E1, E2, two seeds each) and on the follow-cam set;
  - SkiTB tracking (471 SC clips);
  - Ski 2DPose alignment (43/43 segments usable), tracking (43 segments) and propagation (317 labels in 40 segments; reference, tracker and refined crops);
  - the replay (514 clips).
- **The tables step stopped twice.** Both fixes are reporting only; no evaluation output changes.
  1. **Tracking AUC consistency check:** the tolerance goes from 1e-6 to 1e-5 (commit `7715065`).
     - Cause: the track CSVs keep boxes to 3 decimals. 5 of the 87,753 test frames have an IoU within 1e-5 of a success threshold (0.35, 0.45, 0.70 twice, 0.75) and cross it after rounding. The AUC rebuilt from the CSVs is off by 1.6e-6; each crossing moves it by 1/(21 n).
     - Pooled IoU agreed to 2e-10, and the failure-episode counts matched (258).
  2. **Follow-cam pose intervals:** every follow-cam video is a single segment (6 videos, 6 segments; manifest frozen 2026-10-24). D25's bootstrap resamples segments within videos, so each stratum would have one cluster, and `skier.eval.bootstrap` refuses that.
     - **Choice (Ruichao, 2026-11-12):** for the follow-cam set, resample the 6 videos (each one segment). This is the protocol's original video-level bootstrap, which D25 replaced only because test has 3 videos. Code: `pose_tables(..., by_video=True)`, used by `make_tables.py` for `--partition followcam` only.
     - The intervals are wide: 6 units give 462 distinct resamples. The markdown says which unit was used.
     - Chosen for the structure. The follow-cam point estimates had been printed in the run log; no follow-cam interval had been computed.
     - The alternative considered was no intervals, with per-video rows only.
- **Tables and figures** were then built with the same commands as `run_final.sh`, from the commit that records this decision. The commit and the commands are in `$SKIER_RESULTS/final/test/tables_rerun.log`.
- **Effect:** test is spent. Any later change, such as STARK, is reported as an extra experiment, never as a replacement of these numbers.

## D28 — Step 7b: off-the-shelf STARK-ST50 as a labeled post-freeze extra (2026-11-12)
- Choice, recorded before any STARK number exists (Ruichao approved the plan 2026-11-12; work on branch `stark`):
  - **Tracker:** STARK-ST50 `baseline` [yan2021learning], off the shelf.
    - Code: `researchmm/Stark` @ `162e1d37a38f92251cb380e90ccdb59608f2df3c` (MIT) in `$SKIER_DATA/vendor/Stark`.
    - Weights: `checkpoints/train/stark_st2/baseline/STARKST_ep0050.pth.tar` (model-zoo Drive file `1sV_idlYLyxeCIO2o4AQDvUFO5b-X-E4w`; trained on LaSOT, GOT-10k, COCO, TrackingNet), 114,324,945 bytes, sha256 `23c8a677d9c2163e96e5225c1648ac4803e43232aef7c79044a367bde4a86b95`.
    - STARK's own test parameters (`experiments/stark_st2/baseline.yaml`): template 2.0 / 128 px, search 5.0 / 320 px. Dataset name `skitb` is not in STARK's table, so the template update interval is its default, `DATA.MAX_SAMPLE_INTERVAL` = [200] (the same as LaSOT); updates need confidence > 0.5.
    - No STARK setting is tuned. Initialization as for SiamFC: the first visible frame's GT box (SkiTB) or the first label's reference box (Ski 2DPose); no re-init.
    - STARK's code runs unedited under our torch 2.1.2; six import-time snags are handled in `skier.tracking.stark_adapter` (its docstring lists them). Nothing is installed: easydict, which STARK's config imports, was approved but not installed, so a tested attribute-dict stand-in replaces it (the real package is used if present). The backend checks that the parsed config gives the test parameters above and the update interval [200].
  - **Only the tracker changes:** `k_pose` 1.1 and the D26 refinement (`s_min` 0.3, `k_refine` 1.3) stay as chosen for SiamFC; the checkpoints, replay τ/k, splits and bootstrap (D25) are unchanged.
  - **Re-derived for STARK by the D24 rules**, because they depend on the score scale, on STARK's SkiTB val tracks: the confidence signal (raw vs. relative; the higher ROC AUC for failure, IoU < 0.1), the Youden threshold (gate only if AUC ≥ 0.70) and `replay_uses_status` (only if it raises val GT-box containment). They go into `configs/integration/stark.yaml` (`tracker: stark`); `default.yaml` is not edited.
  - **Pre-registered comparison, STARK − SiamFC, paired, D25 units:**
    - SkiTB: pooled mean IoU, success AUC, toolkit F-score (sequence bootstrap); failure-episode counts.
    - Ski 2DPose propagation: E2 and E1, seeds pooled, tracked and refined crops; body and ski PCK@0.1, ski PCK@0.05 (segments within videos).
    - Replay: SkiTB GT-box containment (sequences), Ski 2DPose ski-end containment (segments within videos), crop area as the cost.
  - **Test:** after the val run, the STARK chain runs **once** on test, whatever the val result (no selection on the outcome), from a pushed commit of branch `stark`. It is labeled a post-freeze extra everywhere; SiamFC's D27 numbers stay the main result.
  - **Outputs:** `$SKIER_RESULTS/final_stark/<part>/` (run by `results/step7b/run_stark.sh`); evidence in `results/step7b/`.
  - **Guard:** every new track folder gets a `tracker.json`; a folder without one is SiamFC's. `eval_propagation`, `make_replay --evaluate`, `process_clip` and `make_tables` refuse a tracker that differs from the integration config's `tracker` (default `siamfc`), so STARK tracks can't be gated with SiamFC's raw-scale threshold.
- Reason:
  - SiamFC drift is the largest measured error (D27: tracker crops cost 28.0 ski points on test; 258 failure episodes in 234 of 471 clips). The SkiTB paper ranks off-the-shelf STARK first among the untuned trackers on SkiTB AL (F 0.671; arXiv 2312.09723, Table 9). The milestone listed it as optional.
  - The test was used once (D27), so STARK cannot replace the SiamFC result; a single pre-registered extra run keeps the comparison honest.
- Result on val (2026-11-14):
  - **R1, smoke** (`results/step7b/smoke.json`): the checkpoint loads `strict=True`; boxes are bit-identical on a rerun and with a shared or a fresh backend; 23–28 frames/s with JPEG decoding on the RTX 4070. Clip IoU, STARK vs. SiamFC: AL0027 cam 3 0.850 vs. 0.697; AL0073 cam 7 0.351 vs. 0.005.
  - **R2, SkiTB val tracking:** 75 clips, pooled mean IoU 0.668 (SiamFC 0.363), AUC 0.660. Failure frames (IoU < 0.1, visible, after init): 557 of 12,925 (SiamFC 3,817).
  - **Confidence settings by the D24 rules** (`$SKIER_RESULTS/integration/confidence-val-stark/confidence.json`, in `configs/integration/stark.yaml`):
    - `conf_signal: raw` (ROC AUC 0.829; relative 0.805), `conf_threshold: 0.3258288` (Youden; TPR 0.72, FPR 0.18). 21% of val frames are `low_conf`.
    - `replay_uses_status: true`: GT-box containment 0.992 with it vs. 0.986 without; crop area 42% vs. 38%.
  - **R3, the val chain** (`bash results/step7b/run_stark.sh val`, from the uncommitted branch on top of `9cdcafc`; evidence `results/step7b/val/`): its SkiTB CSVs equal R2's byte for byte; reference-crop hits equal SiamFC's val run exactly; `final/val` unchanged. STARK − SiamFC, paired 95% intervals:
    - SkiTB: pooled IoU 0.668 vs. 0.363 (+0.305 [0.254, 0.336]); AUC +0.295 [0.245, 0.326]; toolkit F 0.681 vs. 0.384 (+0.297 [0.250, 0.328]); failure episodes 6 in 6 clips vs. 46 in 40.
    - Pose, E2 seeds pooled, 261 labels (PCK@0.1): tracker crop body 95.2 vs. 81.1 (+14.1 [6.4, 23.4]), ski 88.0 vs. 75.4 (+12.6 [7.1, 20.2]); refined crop ski 90.9 vs. 81.3 (+9.6 [3.8, 17.1]). Labels with tracker IoU ≥ 0.5: 232 vs. 99; < 0.1: 2 vs. 33.
    - Replay: SkiTB containment 0.992 vs. 0.943 (+0.049 [0.018, 0.074]); crop area 42% vs. 46%. Ski 2DPose ski ends inside 100.0% vs. 99.4%.
    - Not pre-registered, noted: with STARK, refinement still raises ski PCK (88.0 → 90.9) but lowers body slightly (95.2 → 94.7).
- Result on test (2026-11-15, the single run; evidence `results/step7b/test/`):
  - `bash results/step7b/run_stark.sh test` ran once, at `86dd90c` (branch `stark`, pushed; clean tree), 23:41–00:22, exit 0. `freeze.json` (13 files) was written before any output; `final/test` unchanged (tree sha256 `05c6bd8ed440…` before and after). SiamFC's side of every comparison equals D27's tables (e.g. containment 0.963, ski ends 974/1040, 317 labels in 40 segments).
  - SkiTB, 471 clips, 41 sequences: pooled IoU 0.679 vs. 0.404 (+0.276 [0.251, 0.301]); AUC +0.266 [0.242, 0.291]; toolkit F 0.677 vs. 0.406 (+0.270 [0.246, 0.294]), close to the SkiTB paper's off-the-shelf STARK (0.671); failure episodes 66 in 61 clips vs. 258 in 234.
  - Pose, 317 labels, seeds pooled (PCK@0.1): E2 tracker crop body 96.5 vs. 78.9 (+17.6 [9.1, 27.0]), ski 86.5 vs. 67.7 (+18.8 [11.1, 27.8]); refined crop ski 92.1 vs. 77.0 (+15.1 [7.6, 24.1]). E1 shows the same (tracker-crop ski +19.0 [11.5, 27.6]). Labels with tracker IoU ≥ 0.5: 262 vs. 131; < 0.1: 0 vs. 50.
  - With STARK, the tracker crop still costs E2 9.2 ski points [4.9, 15.2] (body 0.5 [−0.3, 1.5]); refinement recovers 5.6 [2.7, 9.0] of them and changes body by −0.1 [−0.9, 0.7].
  - Replay: SkiTB containment 0.990 vs. 0.963 (+0.027 [0.016, 0.040]); crop area 43% vs. 48% (−0.056 [−0.075, −0.034]); Ski 2DPose ski ends inside 99.8% (1038/1040) vs. 93.7% (+6.2 [0.6, 13.2]).
  - Not in the val direction, noted: replay acceleration is higher with STARK on test, 1.69 vs. 1.38 (+0.31 [0.14, 0.49]); on val −0.18 [−0.65, 0.22].
  - These numbers are a labeled post-freeze extra; SiamFC's D27 numbers remain the main result.
