# Project notes: discussion log and next steps

Running record of decisions and open items for the CS230 skier project.
Read this first when resuming work in a new environment.
Companion docs: `skier_proposal.pdf`, `skier_implementation_plan.md`, `skier_step_1_guide.md`,
`skier_step_1_code_spec.md`.

Last updated: 2026-10-17

---

## Working rules (from Ruichao)

- **Updated 2026-10-17:** the repo is shared for planning and coding. Planning
  happens on one machine; coding, testing, downloads, and training happen on
  another, done by Ruichao. Plans must tag each task so it is clear what it needs:
  - `[CODE]` pure coding: no internet, no real data, no GPU; testable with synthetic inputs.
  - `[NET]` needs internet (downloads, installs, cloning, pushing).
  - `[DATA]` needs the real dataset files.
  - `[GPU]` needs a CUDA GPU.
  - `[MANUAL]` human action (forms, accounts, visual review, decisions, report).
- The repo is the **only channel** between the two machines. Anything the planning
  side needs to see (evidence, counts, decisions) must be committed as small text
  files (`results/stepN/`, `docs/`). Dataset images and labels never are.
- (Earlier rule, superseded: Claude does the coding and asks before non-coding tasks.)
- Ruichao is not very familiar with AWS — keep cloud steps minimal and explained.
- **No local installs (2026-10-17):** nothing beyond the light `.venv` is installed on the Mac
  (no torch/mmpose). Torch tests are marked `torch` and run on AWS; first AWS action = full `pytest` (D13).
- **Every step starts with a plan-mode plan**, approved before coding (2026-10-17).
- Manual data inspection is done by Ruichao, using the checklist in
  `skier_step_1_guide.md` §1F (plus §1G step 4 for the video clip).

---

## Review of the existing plans (2026-10-17)

The proposal and both plan files were read and checked against primary sources.
The plans are sound; dataset facts were verified.

### Verified facts

- **Ski 2DPose** (https://www.epfl.ch/labs/cvlab/data/ski-2dpose-dataset/)
  - **No access form.** Direct downloads; only requirement is citing
    Bachmann et al., 3DV 2019. No explicit dataset license text on the page;
    source videos are described as YouTube Creative Commons.
  - Downloads:
    - Labels: https://datasets-cvlab.epfl.ch/2019-ski-2d-pose/ski2dpose_labels.json.zip (5.1 MB)
    - JPG images: https://datasets-cvlab.epfl.ch/2019-ski-2d-pose/ski2dpose_images_jpg.zip (352 MB)
    - (WebP 86 MB / PNG 1671 MB alternatives exist)
    - Videos: https://datasets-cvlab.epfl.ch/2019-ski-2d-pose/ski2dpose_videos.zip (519 MB)
    - Info: https://datasets-cvlab.epfl.ch/2019-ski-2d-pose/ski2dpose_info.json (40 KB)
    - Loader: https://datasets-cvlab.epfl.ch/2019-ski-2d-pose/ski2dpose_example_loader.py
  - 24-joint order and flip pairs in `skier_step_1_guide.md` §1D are **correct**
    (14 body / 8 ski / 2 pole).
  - Visibility: `1` = visible or confidently labeled; `0` = occluded best guess.
  - Coordinates normalized 0–1, origin top-left.
  - JSON nesting: `VideoID → SplitID → ImageID → {annotation: [[x,y,vis]...], frame_idx}`.
    Note: their "SplitID" means a **camera sequence/segment**, not train/val.
    Our code calls it `segment_id`. `frame_idx` is the index within that segment.
  - 1,982 images, 16 videos, 158 sequences. Official split: 147 train / 11 val sequences
    (val = 152 images from 5 video IDs). Images 1920×1080 or 1280×720.
    **Corrected 2026-10-17 from the files:** the page's breakdown (32 SL, 52 GS, 26 SG,
    24 DH, 6 follow-cam) sums to 140, not 158. The info JSON's per-video `type` is the
    source of truth, and follow-cam is **6 separate videos** (see `docs/data_audit.md`).
  - Frames sampled every 0.3–10 s, so leakage risk is same video/course/athlete,
    not near-duplicate frames. Video-grouped splitting is still correct.
- **SkiTB** (https://github.com/matteo-dunnhofer/SkiTB-toolkit)
  - The access form is for the **data only** (frames, boxes, split JSONs).
    Toolkit code is public (CC BY-NC 4.0). SiamFC code is a separate public
    fork (`siamfc-pytorch`); weights `siamfc_alexnet_e50.pth` (GOT-10k
    pretrained, not ski-trained → no test-leakage concern) via a Google Drive
    link in the README.
  - Layout: `AL/ JP/ FS/`, 100 sequences each; per-sequence `frames/`, `MC/`,
    `SC/` (single-camera clips with `boxes.txt`, `frames.txt`, `visibilities.txt`).
    Split JSONs: by date / athlete / course, 2-way and 3-way variants.
  - README uses `--split-file`, code uses `--split_file` (known discrepancy).
- **MMPose 1.3.2** needs mmcv >= 2.0.1 and < 2.2.0, mmengine >= 0.9.0.
  Prebuilt **Windows and Linux** wheels exist for `mmcv==2.1.0` with
  torch 2.1 + cu121, Python 3.8–3.11
  (https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html).

### Local hardware (original machine)

- Windows 10 Pro, **NVIDIA GTX 1070 Ti, 8 GB VRAM**, ~128 GB free on C:.
- No Python installed, no conda, no AWS CLI.
- Repo was in iCloud Drive (`...\iCloudDrive\Desktop\Stanford\2026Fall\CS230\skier_project`).

---

## Proposed changes to the plans (status: APPROVED 2026-10-17, with change 1 replaced by AWS; written into plan v2)

1. **Compute: local GPU first, AWS only as fallback.** Training set is ~1,400
   images; rough unmeasured guess is < 1 h per run on the 1070 Ti at batch 16.
   Step 1C shrinks to: create a Python 3.10 env with pinned
   `torch==2.1.2+cu121`, `mmcv==2.1.0`, `mmengine`, `mmpose==1.3.2`,
   `numpy<2`, plus an env-check script. Use AWS only if measured runtime/VRAM
   is insufficient or to parallelize seeds. Optional now: request an AWS GPU
   (G-instance) vCPU quota increase, since new accounts often start at 0.
2. **Keep data and venv out of iCloud Drive** (sync of ~1 GB data + git folder
   risks corruption). Suggested: data at `C:\skier_data` (or equivalent on new
   machine), configured via a setting/env var, never hard-coded.
3. **Use MMPose only for model construction + COCO checkpoint loading (+ codec);
   write our own dataset, crop transform, masked loss, metrics, and training
   loop in plain PyTorch.** Avoids the ~8 MMPose defaults the plan warns about
   (1.25 bbox padding double-expansion, visibility parsing, loss reduction,
   half-body aug, COCO evaluator, auto LR scaling, flip test, detection files).
4. **Split refinements:** stratify held-out videos by discipline (SL/GS/SG/DH)
   where possible; decide deliberately where the 6 follow-cam sequences go
   (closest proxy for the amateur stretch goal — test or separately reported).
5. **Restructure the Step 1 guide by owner:** Claude writes all scripts (audit,
   inventory, overlays, alignment, smoke tests, env check); Ruichao does the
   SkiTB form, visual review, and any setup they choose to do manually. Step 1A
   should use the existing repo instead of creating a new folder.

---

## Scope discussion: tracking vs. pose

Ruichao suggested doing tracking only first, then pose. Discussion outcome:

- A fixed pretrained tracker + geometric replay has **no learned component**,
  so tracking-only as currently planned likely doesn't meet CS230's
  deep-learning expectation (confirm with TA).
- Pose and tracking are **independent until Step 7 (integration)**: pose
  experiments use reference crops from labels, not tracker boxes.
- Two options:
  - **A (recommended):** keep the plan. Pose fine-tuning is the learning study;
    tracker stays fixed. Prioritize tracking/replay as soon as SkiTB arrives.
  - **B:** make tracking the learned part: fine-tune pretrained SiamFC on SkiTB
    alpine train, add one controlled change (e.g. ski-specific augmentation or
    scale handling), compare pretrained vs fine-tuned vs fine-tuned+change on the
    athlete-disjoint test, plus replay; compare against SkiTB authors'
    skier-focused baseline. Pose becomes a stretch goal. Needs TA OK (changes
    proposal title/question). Do **not** design a tracker from scratch
    (too little data, too big for the budget).
- **Decision: deferred.** Decide when SkiTB access arrives, or by end of
  Week 1 at the latest (B needs fine-tuning time). Pose work started now is
  useful under both options.

---

## Status

- [x] Plans reviewed and verified (2026-10-17)
- [x] **SkiTB access form submitted** (2026-10-17). Follow up if no reply by ~2026-10-24.
      Form: https://forms.office.com/Pages/ResponsePage.aspx?id=Fd5qbmwpJEKsWByOwv1TqGZ0FiFPXjFKh2JII46hSYhUMVQ2RE0yS0FMMEhFSVNWNUxNWks3M0hYSy4u
- [x] **Option A chosen** (2026-10-17): pose fine-tuning is the learning study; tracker stays fixed pretrained SiamFC. May revisit B later.
- [x] **Compute: AWS** (2026-10-17). Ruichao has an AWS budget; training runs on an AWS GPU instance (not the local 1070 Ti). Proposed change 1 is replaced by this.
- [x] Proposed changes approved; MMPose model-only (D3) and split rules (D4) settled (2026-10-17)
- [x] **Plan v2 written** (2026-10-17): `skier_implementation_plan.md` and
      `skier_step_1_guide.md` rewritten with `[CODE]/[NET]/[DATA]/[GPU]/[MANUAL]` tags,
      AWS click-by-click setup, pinned stack, and per-step coding contracts + synthetic tests.
- [x] **Docs re-reviewed; Step 1 code spec written** (2026-10-17, planning machine).
      Fixes: Step 1 evidence now synced to `results/step1/` and committed (was the git-ignored
      `artifacts/`); light dev env + `stack`/`gpu` test markers so `[CODE]` tests run offline;
      plan §10 5.3 notes the loss is the proposal's divided by 64×48; single label-review
      checklist (guide §1F). New `skier_step_1_code_spec.md` is the source of truth for 1A/1B
      code; guide §1B is now a summary table pointing to it.
- [x] **Step 1 offline code written** (2026-10-17, on the planning machine at Ruichao's request):
      scaffold (1A) + all 1B tools per `skier_step_1_code_spec.md`; 71 synthetic tests pass in the
      light env (`.venv`, installed via `uv` from pypi.apple.com because the sandbox proxy blocks PyPI);
      spec §14 acceptance passes. `FILL IN` blocks (spec §13) are still open: they need the real data.
- [ ] **AWS credit requested** (2026-10-17); waiting. No instance or GPU environment yet.
- [x] **Ski 2DPose downloaded, checked and audited** (2026-10-17, Mac, `~/skier_data`):
      checksums in `docs/data_audit.md`; 16 / 158 / 1,982 as expected; **0 audit problems**;
      path template, `info_for` and `locate_segment` filled in; inventory committed at
      `data/manifests/pose_inventory.csv`. Still to do: pilot videos, overlay review (1F), alignment (1G).
- [x] **Step 2 code written** (2026-10-17, after a plan-mode plan): geometry, canonical records,
      split generator (+ `make_pose_split.py`), SkiTB reader (layout unverified), pose and
      tracking metrics; 123 tests pass. Dry run on real data: train 65% / val 16% / test 19%
      + 134-image follow-cam set; perfect-predictor PCK on val = 100%. **Split not frozen yet.**
- [x] **D11 follow-cam = separate evaluation set** (2026-10-17).
- [x] **D10 pilots chosen** (2026-10-17): `5UHRvqx1iuQ`, `oKQFABiOTw8`, `zW1bF2PsB0M`
      (`data/manifests/pilot_videos.txt`). Only these videos are looked at before the freeze.
- [x] **Step 3 code written** (2026-10-17, plan-mode plan approved): heatmaps (D12), preprocessing,
      inference via a plain `heatmap_fn`, E0 evaluation, MMPose cross-check, HRNet wrapper, SiamFC
      adapter + tracking runner. 152 tests pass (2 `stack` tests skip until AWS); mutation check
      4/4 caught; pipeline sanity check on 30 real pilot images within the error bound.
- [x] **Step 4 code written** (2026-10-17, plan-mode plan approved): replay crop controller, renderer
      (replay + debug videos), replay metrics, `make_replay.py` with a val-only tau/k grid. 181 tests
      pass; mutation check 4/4 caught (after adding an uneven-timestep acceleration test). Synthetic
      demo: `~/skier_results/replay/synthetic_demo/out/` (smoothed accel 0.57 vs 4.97 unsmoothed).
- [x] **Step 5 code written** (2026-10-17, plan-mode plan approved; no local installs, D13): augmentation,
      dataset + optional crop cache, masked loss (numpy reference + torch), training loop + `train_pose.py`
      (`--overfit`, `--profile-steps`, `--resume`), `configs/pose/e1.yaml`. 204 tests pass; 6 `torch` + 2
      `stack` tests skip until AWS. Mutation check 5/5 caught. Real pilot samples checked visually.
- [ ] **First on AWS:** full `pytest` (torch + stack tests) → cross-check → (freeze split) → E0 →
      E1 tiny overfit → profile → full E1.

## Next steps

Follow `skier_step_1_guide.md`. In order:

1. `[MANUAL]` Today: AWS budget alert + GPU quota request (guide 1C.1–1C.3).
2. ~~`[CODE]` Offline: implement `skier_step_1_code_spec.md` §2–§11~~ Done 2026-10-17.
3. `[MANUAL][NET]` Launch instance, install pinned stack (1C.4–1C.7).
4. `[NET][DATA]` ~~Downloads (1D), audit (1E)~~ done 2026-10-17 on the Mac. Next: pilot videos,
   overlays + review (1F), alignment (1G) — none need the GPU. Smoke tests (1H) wait for AWS.
5. `[MANUAL]` Readiness decision (1I), sync evidence to `results/step1/` and push, then Step 2.
6. `[MANUAL]` Chase SkiTB if no reply by 2026-10-24.

Rule (2026-10-17): **every step starts with a plan-mode plan, approved before coding.**

To freeze the Step 2 split (in order, before the first AWS scoring run): pilots chosen ✓ → draw overlays and review them (1F) → run
`make_pose_split.py --set-aside followcam=auto` → review summary → commit (freeze), and
review/freeze `docs/evaluation_protocol.md` (draft).

## Manual label review guide (for Ruichao)

Moved to `skier_step_1_guide.md` §1F (legend + 6-point checklist) and §1G step 4
(watching the GT video clip), so there is one copy. Time: ~30–45 min.
