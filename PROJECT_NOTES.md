# Project notes: discussion log and next steps

Running record of decisions and open items for the CS230 skier project.
Read this first when resuming work in a new environment.
Companion docs: `skier_proposal.pdf`, `skier_implementation_plan.md`, `skier_step_1_guide.md`.

Last updated: 2026-10-17

---

## Working rules (from Ruichao)

- Claude does the **coding**. For anything outside coding (installs, downloads,
  account/cloud setup, forms, manual review), **ask Ruichao first** whether they
  want to do it manually before proceeding.
- Ruichao is not very familiar with AWS — keep cloud steps minimal and explained.
- Manual data inspection is done by Ruichao, using a guide Claude provides
  (see "Manual label review guide" below).

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
  - 1,982 images, 16 videos, 158 sequences (32 SL, 52 GS, 26 SG, 24 DH,
    6 follow-cam). Official split: 147 train / 11 val sequences
    (val = 152 images from 5 video IDs). Images 1920×1080 or 1280×720.
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

## Proposed changes to the plans (status: PROPOSED, awaiting approval)

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
- [ ] Option A vs B decision (deferred, see above)
- [ ] Approval of proposed changes 1–5
- [ ] No code written yet; no data downloaded; no environment installed.

## Next steps (pose part, agreed to start now)

Open questions Claude must ask Ruichao before proceeding:

1. Ski 2DPose download (~0.9 GB: labels, JPG, videos, info, loader):
   Claude scripts it (with checksums + provenance log), or Ruichao downloads manually?
2. Python env setup (3.10 + pinned stack, ~3 GB): Claude installs, or
   Claude provides exact commands for Ruichao to run?
3. Repo/data location in the new environment (keep data outside any cloud-sync folder).
4. Approve proposed changes 1–5, and whether to update the two plan files.

Then Claude's coding tasks, in order:

1. Repo scaffold (`docs/ configs/ scripts/ src/ tests/ data/manifests/ results/`),
   `.gitignore` (data, weights, videos, venv), `docs/scope.md`, `docs/decisions.md`.
2. `configs/ski24_schema.json` (names, groups, flip pairs).
3. Label audit script + image inventory → `data/manifests/pose_inventory.csv`.
4. Ground-truth overlay gallery (~30 images, indexed points, L/R colors,
   hollow = occluded, dashed reference box) + contact sheet.
5. Video-frame alignment check for 1–2 pilot segments + short GT clip.
6. Env check script + pretrained COCO HRNet-W32 smoke test on one skier box.
7. Draft `docs/step1_readiness.md`.

## Manual label review guide (for Ruichao)

Purpose: confirm **our code reads the labels correctly** (not re-auditing the
published dataset). For each of ~30 overlays, check:

1. **Position:** points sit on the skier (not in a corner, not consistently shifted).
2. **Left/right:** "right" = skier's own right. Facing camera → their right
   appears on image left. Check front and back views.
3. **Skis:** tip → toes → heel → tail along each ski; toes/heel near the boot.
4. **Poles:** basket point at the bottom end of the pole.
5. **Hollow (occluded) points:** only need to be plausible; note any filled
   point that is clearly wrong.
6. **Box:** dashed box contains whole body and both ski tips.
7. **Video clip:** labels land on the skier in matching frames (not leading/lagging).

Record image ID + issue in `docs/data_audit.md`. If the same issue appears in
more than ~2 of 30 images, report it — it's almost certainly a code bug.
Time: ~30–45 min.
