# Project notes: discussion log and next steps

Running record of decisions and open items for the CS230 skier project.
Read this first when resuming work in a new environment.
Companion docs: `README.md` (setup and commands), `CLAUDE.md` (rules for Claude sessions),
`skier_proposal.pdf`, `skier_implementation_plan.md` (status table at the top), `skier_step_1_guide.md`,
`skier_step_1_code_spec.md`, `docs/decisions.md`.

Last updated: 2026-10-27

---

## Handoff: where things stand (2026-10-26)

Start here in a new session, on any machine. Claude's own notes stay on the machine they were written on, so everything a new session needs is in the repo: this section, `CLAUDE.md`, the decisions log and the plan's status table.

### Done 2026-10-24 to 2026-10-26 (all on the laptop; details under Status)
- **Laptop set up** (D18 pins); the gate passes (now 260 tests).
- **Cross-check passed;** the COCO-checkpoint loading bug fixed.
- **1F overlay review passed;** split, protocol and crop-error set frozen (D19).
- **Runs:** E0, E1, E2–E4 at 50 epochs. The comparison was **confounded by convergence**: the ski tails aren't learned reliably in 50 epochs (`results/step6/diagnostics.md`). D20 trial: 150 epochs fixes it, and seed variance is large.
- **D21 rerun done (2026-10-27 00:19):** E1–E4 at 150 epochs × seeds 230/231. On the seed means, D16's rule picks **E2** (crop jitter): val ski PCK@0.1 93.1 vs. E1 91.9; both seeds agree. Recipe choice is Ruichao's (`results/step6/d21/`).
- **Tracker set up:** toolkit and SiamFC fork cloned, weights downloaded, the adapter's `FILL IN` block done.
  - **Smoke test** on pilot `5UHRvqx1iuQ/0`: it tracks for about 3 s, then drifts to the crowd and locks onto the broadcast logo.
- **1G alignment done:** labels match their clip frames exactly, offset 0 (`docs/data_audit_ski2dpose.md` §1G).
- **SkiTB frame rate:** the toolkit doesn't have it either (`docs/data_audit_skitb.md` finding 2).

### Done 2026-10-18 to 2026-10-24
- **SkiTB:** the alpine set (`AL/`) arrived and is audited (D15, `docs/data_audit_skitb.md`), and its reader is filled in. The data-audit doc is split per dataset.
- **Step 6 code** (D16): augmentations with paired draws, E2–E4 configs, the crop-error set, the checkpoint evaluator, the comparison table with the pre-registered selection rule, and the augmentation preview.
- **AWS:** credits redeemed, budget alert set, key pair `skier-aws` created in us-west-2. The key is at `~/.ssh/skier-aws.pem` on the Mac. **The account is still being verified**, so nothing is launched (see Status and "AWS lessons" below).
- **D17:** GPU work moves to Ruichao's Windows laptop (RTX 4070, WSL2), following `docs/laptop_setup.md`. AWS is kept for later.

### Machine state

| Machine | State |
| --- | --- |
| **Mac** | Repo at `~/Desktop/skier_project`; light `.venv` (250 passed, 9 skipped). Data in `~/skier_data`: Ski 2DPose, plus SkiTB (`AL/FS/JP`; only `AL` is used). Results in `~/skier_results`: `step1/audit`, `pose/aug_preview/e1–e4.png`, `replay/synthetic_demo`, `skitb_audit`. The 1F overlays were drawn and reviewed on the laptop instead (2026-10-24). No `skier-aws` entry in `~/.ssh/config` yet. |
| **Laptop (2026-10-26)** | `~/skier_data`: `ski2dpose`, `pretrained` (COCO HRNet), `vendor/` (SkiTB toolkit `b402f9a`, SiamFC fork `a87f140` + `siamfc_alexnet_e50.pth`). **SkiTB: all of `skitb/` copied and verified (2026-10-26); only `AL/` is used.** `~/skier_results/pose/`: E0–E4 (50 epochs), the D20 runs `E1-seed231` and `E3-ft150`, and `d21/` (the rerun: 8 runs, evaluations and comparisons, done 2026-10-27). `~/skier_results/tracking/skitb-val/` and `~/skier_results/replay/` (grid and review set). The WSL clone may carry uncommitted copies of the Windows clone's edits (configs etc.); `git checkout -- <files>` before `git pull`. Windows: sleep on AC is Never; **Windows Update isn't paused**, and it restarted the laptop on 2026-10-24. History: **Parts 1–2 done (2026-10-24).** WSL2 Ubuntu 22.04.5, Linux user `ruichaoz`; venv `~/venvs/skier` with the pinned stack plus D18's two pins; repo cloned at `~/skier_project` (GitHub SSH key added); `SKIER_DATA`/`SKIER_RESULTS` set in `~/.bashrc`. **Gate passed: 257 passed, `check_env --strict` exit 0**; `env.json` and `pip-freeze.txt` in `~/skier_results/step1/env/` (copies in `results/step1/laptop/`). **Parts 3–4 done (2026-10-24):** Ski 2DPose in `~/skier_data/ski2dpose` (all 4 SHA-256 match the audit; 16 / 158 / 1,982), COCO HRNet-W32 config and checkpoint in `~/skier_data/pretrained`; pose smoke test OK (54.7 ms/image, 247 MB); **MMPose cross-check PASS** (mean 0.014 px, max 1.19 px; `results/step3/crosscheck.json`). Gate now 259 passed. No SkiTB, SiamFC or tracker yet. **Later the same day:** 1F overlays reviewed, split frozen (D19), E0 on val, E1 overfit (standard and a 300-epoch diagnostic) and the E1 profile run; see Status. Full E1 not started. |
| **AWS** | Account verification pending: the GPU-quota request was auto-rejected, the Standard quota is 1 vCPU, and CloudShell says "account verification is in progress". No instance yet. |

### Waiting on Ruichao, in priority order
1. ~~**Laptop setup and trial** (`docs/laptop_setup.md` Parts 1–4).~~ Done 2026-10-24, cross-check passed.
   - A Claude session can also run on the laptop's **Windows side** (repo clone in `OneDrive\Desktop\skier_project`, used only for editing). It drives Ubuntu with `wsl -d Ubuntu-22.04 -- bash <script>`. Multi-line commands go in a script file, because PowerShell mangles `$` quoting. Root without a password: `wsl -d Ubuntu-22.04 -u root`, used once for the apt packages of 2.2.
2. ~~**Overlay review (1F), then freeze the split, then the crop-error set**~~ All done 2026-10-24 on the laptop (D19; crop-error set `1e9c0e1c9e7e…`).
3. ~~**Copy SkiTB to the laptop**~~ Done and verified 2026-10-26. Optional: delete the unused `FS/` and `JP/` there.
4. ~~**The SkiTB frame rate of each sequence**~~ Settled as D22 (2026-10-26).
5. **Choose the recipe:** the D21 seed-mean comparison is ready (`results/step6/d21/compare_val_seedmean/comparison.md`); D16's rule suggests E2.
6. ~~**SiamFC tracker class** and the **1G alignment check**~~ Done 2026-10-26.
7. **Pause Windows Update** before long runs.
8. **AWS, after verification:** reapply for the quotas: G and VT at 4 vCPUs in us-west-2 (and us-east-1), Standard at 8. If the account is on AWS's "Free plan", upgrading to the Paid plan keeps the credits and may be required for GPUs; that's Ruichao's call.

### AWS lessons (so they aren't rediscovered)
- **"Global" in the region menu** means an account-wide page (Billing, IAM, Budgets, EC2 Global View). EC2, key pairs and quotas are per region, so use us-west-2: https://us-west-2.console.aws.amazon.com/ec2/home?region=us-west-2
- **Quotas:** requests are free. Until account verification finishes, requests are rejected automatically and quotas sit at placeholder values. The relevant quota is "Running On-Demand G and VT instances", measured in **vCPUs** (a g5.xlarge needs 4).
- **Credits** come off the monthly bill automatically. To confirm, look for a **Credits** line in Billing → Bills; it can take about 24 h to appear. Check the credits' expiry and eligible services. Budget alerts may measure cost after credits.
- **Trial path once verified:**
  - Run an `m5.large` CPU instance with the Deep Learning Base GPU AMI (Ubuntu 22.04), then later **stop it → change instance type → `g5.xlarge`**, keeping the same disk.
  - Within a 1-vCPU limit: `t2.small` with plain Ubuntu works. `t2.micro` doesn't: it's free-tier, so it wouldn't test the credits, and its 1 GB of RAM is too little.
- **Habits:** stop the instance when idle and never terminate it (that deletes the disk). The public DNS changes after every stop/start.

### How Ruichao likes to work (in addition to the working rules)
- Step-by-step walkthroughs for console and infrastructure tasks. Ruichao runs the commands in their own terminal and pastes the output.
  - **Exception (2026-10-24):** when Claude runs on the laptop's Windows side, Ruichao wants Claude to **run the commands itself** through `wsl -d Ubuntu-22.04 -- bash <script>`. Stop only for what needs Ruichao: GitHub web settings, passwords, approvals, visual reviews, and copying data from the Mac.
- **Long runs:** start them in tmux inside Ubuntu, with an Ubuntu window left open. Interrupting a Claude session that holds the only `wsl` connection shut WSL down once (2026-10-25).
- A cheap trial run before any compute-heavy work.
- Keep options open: don't drop plans (AWS, experiments). Move doable work to whatever machine is available.
- When Ruichao says "don't do anything yet, just tell me", answer only.

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
    source of truth, and follow-cam is **6 separate videos** (see `docs/data_audit_ski2dpose.md`).
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
  - **Data received 2026-10-18**, now at `~/skier_data/skitb/`. Layout confirmed (`SC/`, bare frame
    indices → `NNNNN.jpg`); audit in `docs/data_audit_skitb.md`. The athlete split shares 5 athletes
    between train and test (no leakage: nothing is trained on SkiTB). The frame rate is not in the files
    and differs by sequence (about 25 or 30 fps).
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
- **Decision: A** (D1, 2026-10-17). It had been deferred until SkiTB arrived; **confirmed
  2026-10-18** when it did (D15).

---

## Status

- [x] Plans reviewed and verified (2026-10-17)
- [x] **SkiTB access form submitted** (2026-10-17). **Approved; data received 2026-10-18.**
      Form: https://forms.office.com/Pages/ResponsePage.aspx?id=Fd5qbmwpJEKsWByOwv1TqGZ0FiFPXjFKh2JII46hSYhUMVQ2RE0yS0FMMEhFSVNWNUxNWks3M0hYSy4u
- [x] **Option A chosen** (2026-10-17): pose fine-tuning is the learning study; tracker stays fixed pretrained SiamFC. Confirmed 2026-10-18 (D15).
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
- [x] **AWS credits redeemed**, budget alert and key pair `skier-aws` created (us-west-2). **Account verification still
      in progress** (2026-10-24): the GPU-quota request was rejected automatically, the Standard quota is 1 vCPU, and
      CloudShell won't start. Reapply (G and VT: 4 vCPUs; Standard: 8) once verification completes.
- [x] **D17 (2026-10-24): GPU work moves to Ruichao's Windows laptop** (RTX 4070, WSL2) for now; AWS kept for
      heavier or parallel work later. Setup note: `docs/laptop_setup.md`.
- [x] **Ski 2DPose downloaded, checked and audited** (2026-10-17, Mac, `~/skier_data`):
      checksums in `docs/data_audit_ski2dpose.md`; 16 / 158 / 1,982 as expected; **0 audit problems**;
      path template, `info_for` and `locate_segment` filled in; inventory committed at
      `data/manifests/pose_inventory.csv`. Still to do: pilot videos, overlay review (1F), alignment (1G).
- [x] **Step 2 code written** (2026-10-17, after a plan-mode plan): geometry, canonical records,
      split generator (+ `make_pose_split.py`), SkiTB reader (layout confirmed 2026-10-18), pose and
      tracking metrics; 123 tests pass. Dry run on real data: train 65% / val 16% / test 19%
      + 134-image follow-cam set; perfect-predictor PCK on val = 100%. **Frozen 2026-10-24 (D19).**
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
- [x] **README.md and CLAUDE.md written; plan status table added** (2026-10-18).
- [x] **Deadline moved up to Fri 2026-11-20** (2026-10-18, D14): plan §4 rewritten as 5 weeks; fallbacks dated.
- [x] **SkiTB received and audited** (2026-10-18, D15): `AL/` at `~/skier_data/skitb/`; 0 format problems;
      100 sequences, 1,100 single-camera clips (train 554 / val 75 / test 471). The data-audit doc is split into
      `docs/data_audit_ski2dpose.md` and `docs/data_audit_skitb.md`; evidence in `results/step1/skitb_audit/`.
      The reader's layout block and `frame_file` are filled in; 205 tests pass. **Open:** the frame rate of each sequence.
- [x] **Step 6 code written** (2026-10-18, plan-mode plan approved; D16): crop jitter, motion blur and occlusion with a
      paired design (E1–E4 share flips, rotations, scales and data order); `configs/pose/e2–e4.yaml`; crop-error set
      generator; checkpoint evaluator (clean, strata, crop-error, cut-off share); comparison table with the
      pre-registered selection rule; augmentation preview; `train_pose.py --seed`. 249 tests pass; mutation check
      10/10 caught. Pilot augmentation sheets checked visually (`~/skier_results/pose/aug_preview/`).
- [x] **Laptop set up and gate passed** (2026-10-24, `docs/laptop_setup.md` Parts 1–2): 257 passed, `check_env --strict` OK on the
      RTX 4070 (8 GB). Two install fixes, recorded as D18: `mmdet==3.2.0` (mmpose can't load models without it) and
      `opencv-python<4.12` (mmcv pulled OpenCV 5.0, which needs numpy ≥ 2).
- [x] **Ski 2DPose, COCO weights, smoke test and MMPose cross-check on the laptop** (2026-10-24, Parts 3–4). Checksums match;
      cross-check **PASS**, mean 0.014 px, max 1.19 px over 10 pilot images (9 of 10 identical). First run failed: the COCO
      checkpoint carries `data_preprocessor.mean/std`, which `PoseNet` lacks, and `build_hrnet`'s strict load check refused it.
      Fix in `pose/model.py`: those two tensors are checked against `preprocess.MEAN/STD` (they match) and then left out.
      +1 light, +1 `stack` test; mutation check 3/3 caught. This also unblocks `train_pose.py`, which loads the same checkpoint.
- [x] **Overlay review (1F) passed; split and evaluation protocol frozen** (2026-10-24, D19). Ruichao reviewed 30 pilot
      overlays drawn on the laptop: all six checks OK. `data/manifests/pose_split_v1.json`, SHA-256 `46bd127b0e51…`:
      train 1,194 / val 293 / test 360 images (65/16/19), followcam 134; identical to the 2026-10-17 dry run.
- [x] **E0, E1 overfit and E1 profile on the laptop** (2026-10-24; evidence in `results/step3/`, `results/step5/`):
      - **E0 on val:** body (10 shared joints) PCK@0.1 **95.1%** (2,517 / 2,647), PCK@0.05 85.2%; ski N/A.
      - **Overfit, standard (`--overfit 32`, 53 epochs = 106 steps):** body 97.2%, ski only 64.3%. In the overlays the ski
        predictions sit on the boots: an undertrained new head, not a pipeline bug.
      - **Overfit, diagnostic (`--epochs 300`):** body 100% by epoch 75, **ski 100% by epoch 200**, loss 0.0040 → 0.00019.
        The pipeline is verified for ski points too. Ski PCK plateaus near 61% for epochs 10–50, then climbs.
        **A future overfit check needs about 300 epochs**; 53 is too short at 2 steps per epoch.
      - **Profile:** the default `--profile-steps 200` measures only warm-up steps (75 steps per epoch, 3 frozen epochs), giving
        0.057 s/step and 0.34 GB. Profiled with `warm_epochs: 0` instead (`results/step5/profile_finetune_config.yaml`), the
        fine-tune phase takes **0.171 s/step and 2.9 GB peak of 8 GB**, with 5% data wait; estimated full E1 **0.21 h**.
        **Batch 16 fits; no batch-size change is needed** (D17). Ruichao chose to keep this as a documented workaround rather
        than change `--profile-steps` in `train.py`, so E1–E4 all run on the same training code.
- [x] **Crop-error set frozen** (2026-10-24, before any E1 result; Ruichao approved): `data/manifests/crop_error_set_v1.json`,
      D16 defaults (val, seed 230, levels 0.05 / 0.10 / 0.20), 293 val images, content SHA-256 `1e9c0e1c9e7e…`, tied to split
      `46bd127b0e51…`. It passes `check_crop_error_set`, and the tool refuses to overwrite it.
- [x] **Full E1 trained** (2026-10-24, laptop, `configs/pose/e1.yaml` unchanged, commit `4de9902`; `results/step5/E1_*`):
      best epoch 47 by val ski PCK@0.1: **ski 89.50% (1,568 / 1,752), body 96.89% (3,642 / 3,759)**. Ski PCK climbs from
      about 47% after warm-up to about 89% by epoch 32, then stays flat. Re-validating `best.pt` reproduces 89.50% exactly.
      Val overlays: ski lines follow the labels even in fog and motion blur. Pole baskets are often placed at the hands or
      skis (supplementary only, D7), and left/right overlap in side views.
      - **Run time:** 12.9 min of compute (0.17 s/step, 2.9 GB peak, as profiled), but **34.8 min wall clock**: Windows
        entered **Modern Standby from 19:20:25 to 19:42:20** (Kernel-Power events 506/507), which paused WSL. The run resumed
        unharmed. Keep the laptop from sleeping during runs (`docs/laptop_setup.md` Part 6).
- [x] **E2–E4 trained, E1–E4 evaluated on val, comparison run** (2026-10-24; `results/step5/E{2,3,4}_*`, `results/step6/`).
      A Windows Update restart (20:31:40, `MoUsoCoreWorker.exe`) stopped E4 after epoch 11; it resumed from `last.pt` (config
      and split checked) and finished. Clean val ski PCK@0.1: **E1 89.5, E2 69.3, E3 77.4, E4 77.6**; body 96.1–97.9 for all.
      **D16's rule picks E1** (no candidate beats it). **Recipe choice not made yet (Ruichao).**
      - **Diagnosis (`results/step6/diagnostics.md`): convergence, not augmentation.** The whole ski deficit is in ski
        **tails** (E2–E4, about 12–18% vs 65–77%) and, in E2 only, **tips** (64–66% vs 91–94%). E2–E4 put tails at the boot
        **even on their own training images** (about 20% vs E1's 73–78%), although E3's training targets are bit-identical
        to E1's. E2 and E4 share crops and jitter, yet only E2's tips fail. So whether a run learns the ski ends within
        50 epochs (×0.1 at epochs 30 and 42) varies by run; the table doesn't measure augmentation.
- [x] **D20 convergence trial** (2026-10-25; `results/step6/diagnostics.md` §5, `results/step6/trial_d20/`). The reading was
      fixed in advance. **E1-seed231 fails on the tails** (training images about 20%; val ski 69.4%), so plain E1 fails under
      another seed. **E3-ft150 (150 fine-tune epochs) learns them** (95 / 80%; val ski 91.8%, best epoch 144). D20's row
      "E1-seed231 fails, E3-ft150 learns" → **rerun E1–E4 with the longer schedule and 2+ seeds per recipe (new decision
      needed; Ruichao).** The session interruption at 19:17 shut WSL down; E1-seed231 resumed from epoch 8.
- [x] **D21 rerun done** (2026-10-26 19:44 → 2026-10-27 00:19, Ruichao approved; `results/step6/d21/`): all four configs at
      `finetune_epochs: 150` (also the code default, so `e1.yaml` stays the default); seeds 230/231; E3-ft150 reused as
      `E3-s230` (config, seed and split checked identical). 7 new runs, about 38–40 min each, no interruptions; all 153
      epochs logged; all 8 evaluated on val (same split, samples and crop-error set; denominators identical).
      - **Seed pooling** added to `compare.py` (`E1=s230.json,s231.json`): hits and totals summed, totals must match, seeds
        must be distinct and the same for every run; per-seed ski PCK shown next to the mean. 4 new tests, mutation check
        11/11; 268 tests pass on the laptop.
      - **Seed means, clean val PCK@0.1** (body / ski / ski on moderate crop errors):
        E1 95.2 / 91.9 / 90.4; **E2 95.5 / 93.1 / 92.9**; E3 96.9 / 92.0 / 90.9; E4 97.1 / 91.0 / 89.5.
        Per seed (230 / 231), ski: E1 92.2 / 91.6, E2 92.5 / 93.7, E3 91.8 / 92.3, E4 90.8 / 91.1.
      - **D16's rule on the seed means picks E2** (crop jitter): nobody is disqualified on body; E2 and E3 beat E1;
        E2 − E3 = 1.08 points, just outside the 1-point tie band (the tie-break would pick E2 too: 92.9 vs 90.9).
        **Each seed on its own also picks E2.**
      - **Where it comes from:** E2's gain is in the ski **tails** (seed means +10.9 left, +7.4 right vs. E1); tips and the
        body are about even. Blur and occlusion (E3, E4) raise most body joints by 1–4 points but lose 8–10 points on the
        left tail. E4 (all three) has the lowest ski PCK@0.05 (78.8 vs. 82.9).
      - The severe-level cut-off share is 9.4% for every run, as it should be (it depends only on the crops).
      - **Convergence check** (`results/step6/trial_eval.py --out=…`, `results/step6/d21/train_fit_d21.{txt,json}`): **all 8
        runs learn the tails** on training images (D20 threshold 60%), so the convergence confound is gone. Training-image
        tails (right / left): E1 93 / 96 and 87 / 86; E2 99.6 / 99.9 and 99.4 / 98.7; E3 95 / 80 and 94 / 84;
        E4 80 / 81 and 83 / 78. E3 and E4 still fit the tails less well than E1 and E2, even on training images, so part
        of their tail deficit may be slower fitting under blur and occlusion rather than worse generalization. That is a
        caveat for the report; the pre-registered rule is unchanged.
      - **Recipe choice: Ruichao** (record it in `docs/decisions.md`). Bootstrap intervals come in Step 8.
- [x] **Tracker set up and smoke-tested** (2026-10-26, laptop; `docs/data_audit_skitb.md` 1D table):
      - SiamFC adapter `FILL IN` filled in from the fork's source: `siamfc.TrackerSiamFC`, with `return_conf=True` giving
        `(box, conf)`. Box conventions round-trip exactly. A new torch-free test with a fake fork module catches 3/3
        planted bugs; the suite has 260 passed.
      - **Smoke test** (`5UHRvqx1iuQ/0`, started on labelled frame 70): it tracks for about 3 s (IoU vs. label boxes 0.67
        at frame 79, 0.40 at 129), then **drifts to the course crew and locks onto the static broadcast logo** (frames 180+).
        That's a fixed-tracker failure, not an adapter bug (stills checked). Step 7 should plan for it.
- [x] **1G alignment done** (2026-10-26): `5UHRvqx1iuQ/0` and `oKQFABiOTw8/22`, all 94 images at offset 0, NCC ≥ 0.9999.
      `frame_idx` = 0-based clip frame. `MIN_SCORE` 0.99, `MIN_MARGIN` 0.02 (`docs/data_audit_ski2dpose.md` §1G).
- [x] **SkiTB on the laptop, verified** (2026-10-26): Ruichao copied all of `skitb/` (AL, FS, JP) into WSL. 363,486 macOS
      `._*` sidecars removed (all verified AppleDouble). All 100 AL sequence digests and the tree digest match the Mac
      audit (`docs/data_audit_skitb.md`). `FS/` and `JP/` (7.4 GB) are unused; deleting them is Ruichao's call.
- [x] **D22: per-sequence SkiTB frame rate** (2026-10-26, approved). The paper says 25/30 FPS. The estimate (MC span ÷ race
      time) snaps cleanly to 18 × 25 and 82 × 30 (`data/manifests/skitb_fps_v1.csv`, frozen). `run_tracker.py` and
      `make_replay.py` use it by default; `--fps` is an override. 264 tests pass; mutation check 5/5.
- [x] **Tracker on SkiTB val** (2026-10-26, 75 clips, 12,925 frames, 3.5 min; `results/step3/skitb_val_tracking_summary.json`).
      Fixed pretrained SiamFC, D22 frame rates. **Pooled mean IoU 0.363, AUC 0.364**; success at IoU 0.1/0.3/0.5/0.7 =
      0.71/0.59/0.37/0.13. Per-clip mean IoU: 12% of clips below 0.1, 33% at or above 0.5. **46 failure episodes** (IoU < 0.1
      for ≥ 0.5 s) in 40 of 75 clips; median 1.6 s, max 9.5 s. By sequence, frame-weighted IoU ranges from 0.25 (AL0026) to
      0.55 (AL0044). These are our metrics; the toolkit's official ones are separate (Step 8).
      **Step 7 consequence:** pose crops from these boxes will often miss the skier. Plan the integration around that
      (score-based gating, reporting tracker-crop vs. reference-crop pose separately).
- [x] **Replay τ/k grid on SkiTB val** (2026-10-26, 75 clips, 8 s; `results/step4/grid_val.csv`). GT-box containment is
      0.82–0.87 everywhere. It depends on k, not τ, and is capped by tracker failures. Smoothing cuts crop acceleration
      3–4× at equal containment: τ 0.2, k 2.0 gives containment 0.841, acceleration 1.65 vs. 5.49 unsmoothed, crop area 28%
      of the frame. τ 0.4 raises centering error and, at small k, p95 acceleration.
      **Review set for Ruichao:** `$SKIER_RESULTS/replay/review/`. Side-by-side smoothed vs. unsmoothed, plus the debug
      view, for AL0027 cam 3 (well tracked, IoU 0.70) and AL0073 cam 7 (tracker lost within 5 frames, IoU 0.005).
      **Defaults not chosen yet (Ruichao, plan §9 review block).**
- [x] **Tracker-upgrade check** (2026-10-27, look only; nothing installed). Order agreed with Ruichao: Step 7 core with
      SiamFC (with low-confidence flagging) first; an optional, time-boxed tracker upgrade after it, chosen on val;
      then Step 8 freezes the tracker.
      - **STARKski** (the SkiTB authors' tracker): two STARK-ST50 copies **fine-tuned on SkiTB**; when the main one's
        confidence drops to 0.5 or below, a second copy with a search area the height of the frame re-detects the skier.
        **No public weights found** (toolkit README, arXiv 2312.09723, arXiv 2502.18867).
      - **Off-the-shelf trackers on SkiTB AL, single-camera, F-score** (arXiv 2312.09723 Table 9): STARK 0.671,
        CoCoLoT 0.652, KeepTrack 0.640, MixFormer 0.637, LTMU 0.623, OSTrack 0.594, GlobalTrack 0.585, SiamRPN++ 0.536.
        SiamFC isn't in the table. Fine-tuned trackers: STARK_FT 0.875, STARKski 0.882, YOLO-SORT 0.819. The F-score is the
        toolkit's metric, not our IoU/AUC.
      - **Candidate: off-the-shelf STARK-ST50** (`researchmm/Stark`, MIT). Its weights are in the repo's model zoo. Its
        API fits our adapter: `STARK_ST.initialize(image, {"init_bbox": [x, y, w, h]})`, then `track(image)` →
        `{"target_bbox": [x, y, w, h], "conf_score": float}`. STARKski's re-detection works at inference time, so it
        can be copied without training. The install script targets Python 3.6 / PyTorch 1.7, so it needs its own venv
        (keeping the pinned pose stack untouched) and Ruichao's OK.
- [ ] **First GPU session (laptop, `docs/laptop_setup.md`):** ~~full `pytest` (torch + stack tests)~~ done → data (Ski 2DPose; the SkiTB val/test subset)
      → cross-check → (freeze split) → E0 → E1 tiny overfit → profile → full E1 → (freeze the crop-error set)
      → E2–E4 → evaluate and compare on val.

## Next steps

Current order (2026-10-24):

A. ~~`[MANUAL][NET]` **Laptop setup and trial** (`docs/laptop_setup.md` Parts 1–4).~~ Done 2026-10-24; cross-check passed.
B. ~~`[MANUAL]` **Overlay review (1F) → freeze the split → the crop-error set**~~ All done 2026-10-24 (D19; crop-error set above).
C. `[GPU]` On the laptop: ~~E0 → E1 overfit → profile → full E1 → E2–E4 → evaluate → compare~~ done 2026-10-24.
   ~~Decide how to handle the convergence confound~~ → D20 trial done 2026-10-25. ~~Approve the rerun (D21)~~ done
   2026-10-27; the rule picks E2 on the seed means. **Next: Ruichao chooses the recipe; then Step 7 (plan mode).**
D. `[MANUAL]` AWS: wait for account verification, then reapply for the quotas. Not blocking anything now (D17).

The original Step 1 order, kept for reference (follow `skier_step_1_guide.md`):

1. `[MANUAL]` ~~AWS budget alert + GPU quota request (guide 1C.1–1C.3).~~ Done; quota pending verification (see Status).
2. ~~`[CODE]` Offline: implement `skier_step_1_code_spec.md` §2–§11~~ Done 2026-10-17.
3. `[MANUAL][NET]` Launch instance, install pinned stack (1C.4–1C.7).
4. `[NET][DATA]` ~~Downloads (1D), audit (1E)~~ done 2026-10-17 on the Mac. Next: pilot videos,
   overlays + review (1F), alignment (1G) — none need the GPU. Smoke tests (1H) wait for AWS.
5. `[MANUAL]` Readiness decision (1I), sync evidence to `results/step1/` and push, then Step 2.
6. ~~`[MANUAL]` Chase SkiTB if no reply by 2026-10-24.~~ Received 2026-10-18.
7. `[MANUAL][NET]` Find SkiTB's frame rate for each sequence (toolkit source, or the source videos'
   metadata; `docs/data_audit_skitb.md` finding 2). Then the tracker and replay take it per clip.
8. `[MANUAL]` Optional: look at the augmentation sheets (`~/skier_results/pose/aug_preview/e1–e4.png`).
9. `[CODE]` Step 7 (integration): plan after the alignment check (1G) and the first AWS session.

Rule (2026-10-17): **every step starts with a plan-mode plan, approved before coding.**

The Step 2 split is frozen (2026-10-24, D19): pilots chosen ✓ → overlays reviewed (1F) ✓ →
`make_pose_split.py --set-aside followcam=auto` ✓ → summary reviewed ✓ → committed ✓, with `docs/evaluation_protocol.md` ✓.

## Manual label review guide (for Ruichao)

Moved to `skier_step_1_guide.md` §1F (legend + 6-point checklist) and §1G step 4
(watching the GT video clip), so there is one copy. Time: ~30–45 min.
