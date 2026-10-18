# Skier tracking and 2D body-and-ski landmark estimation

CS230 project, Fall 2026 · Ruichao Zhang (ruichaoz@stanford.edu)

Given a single-camera skiing video and a box around the skier in the first frame, the system:
- tracks the skier with a fixed, pretrained SiamFC;
- produces a smooth, centered 16:9 replay;
- predicts **24 landmarks** (14 body, 8 ski, 2 pole) with a fine-tuned COCO HRNet-W32.

The learning study asks whether targeted augmentation (crop jitter, motion blur, occlusion) improves ski-landmark accuracy and robustness to imperfect crops. Proposal: `skier_proposal.pdf`. Plan: `skier_implementation_plan.md`.

```mermaid
flowchart LR
    A["Video + first-frame box"] --> B["Fixed SiamFC tracker"]
    B --> C["Pose crop"] --> D["Fine-tuned 24-point HRNet"]
    B --> E["Smoothed 16:9 replay crop"]
    D --> F["Landmarks drawn into the replay"]
    E --> F
```

## Status (2026-10-18)

**Target completion: Friday 2026-11-20.** The week-by-week schedule is in plan §4.

| Step | Code | Real runs |
| --- | --- | --- |
| 1 Setup and data checks | ✅ done | ✅ Ski 2DPose and SkiTB `AL/` downloaded and audited (0 problems). ⏳ Overlay review, alignment check, GPU smoke tests |
| 2 Split, geometry, metrics | ✅ done | ⏳ Split freeze (after the overlay review) |
| 3 Baselines (E0, tracker) | ✅ done | ⏳ AWS |
| 4 Centered replay | ✅ done | ⏳ Needs tracker output (AWS) |
| 5 Train 24-point model (E1) | ✅ done | ⏳ AWS |
| 6 Augmentation study (E2–E4) | ✅ done | ⏳ AWS, after the split freeze. ✅ Augmentation preview on pilot images |
| 7–8 Integration, report | not started | — |

- The **AWS** GPU instance is pending.
- **SkiTB** data arrived 2026-10-18 and is audited. The frame rate of each sequence is still open.
- The test suite has 249 passing tests; 8 more need PyTorch or mmpose and run on AWS.
- Detailed status and next actions: `PROJECT_NOTES.md`.

## Setup

**Light environment (any machine; enough for every unit test):**

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # numpy<2, opencv-python, pillow, pyyaml, pytest
pytest -q                        # torch/stack/gpu tests skip without those packages
```

**GPU environment (AWS g5.xlarge):** the pinned stack is torch 2.1.2 + cu121, mmcv 2.1.0, mmengine 0.10.x, mmpose 1.3.2, numpy < 2.
- Install it exactly as in `skier_step_1_guide.md` §1C.7; the pins are listed in `requirements-pose.txt`.
- Then run the full test suite, **before any training**.
- `python scripts/check_env.py --strict` must pass.

**Paths:** data and results live outside the repo. Set `SKIER_DATA` (default `~/skier_data`) and `SKIER_RESULTS` (default `~/skier_results`).

## Data

**Ski 2DPose** (Bachmann et al., 3DV 2019): public, no form needed.

```bash
mkdir -p $SKIER_DATA/ski2dpose/archives && cd $SKIER_DATA/ski2dpose/archives
B=https://datasets-cvlab.epfl.ch/2019-ski-2d-pose
for f in ski2dpose_labels.json.zip ski2dpose_images_jpg.zip ski2dpose_videos.zip ski2dpose_info.json; do curl -fLO "$B/$f"; done
cd .. && mkdir labels images videos
unzip -q archives/ski2dpose_labels.json.zip -d labels
unzip -q archives/ski2dpose_images_jpg.zip -d images
unzip -q archives/ski2dpose_videos.zip -d videos
```

- Compare the SHA-256 values with `docs/data_audit_ski2dpose.md`.
- The expected counts are 16 videos, 158 segments and 1,982 images.

**SkiTB** (Dunnhofer et al., WACV 2024; CC BY-NC 4.0): access by form. Only the alpine (`AL/`) data is used.
- Put it at `$SKIER_DATA/skitb/` (so `$SKIER_DATA/skitb/AL/AL0001/…`) and pass `--skitb-root $SKIER_DATA/skitb`.
- The split file is `AL/AL_train_val_test_athlete_60-40.json` (D15).
- Layout, digests and findings: `docs/data_audit_skitb.md`.

Dataset images and labels are **never** committed. The repo holds only manifests and small text evidence.

## Running the pipeline

Every tool has `--help`. Outputs go under `$SKIER_RESULTS` unless `--out` is given. `L` stands for `$SKIER_DATA/ski2dpose/labels/ski2dpose_labels.json`, and `I` for `$SKIER_DATA/ski2dpose/images`.

| Stage | Command |
| --- | --- |
| Label audit and inventory | `python scripts/audit_pose_labels.py $L --info $SKIER_DATA/ski2dpose/archives/ski2dpose_info.json --images-root $I` |
| Ground-truth overlays (pilot videos) | `python scripts/draw_gt_overlays.py --inventory data/manifests/pose_inventory.csv --labels $L --images-root $I --videos-file data/manifests/pilot_videos.txt` |
| Video-frame alignment | `python scripts/check_alignment.py --inventory data/manifests/pose_inventory.csv --images-root $I --video-id VID --segment-id SEG --labels $L` |
| **Freeze the pose split** | `python scripts/make_pose_split.py --set-aside followcam=auto` → `data/manifests/pose_split_v1.json` |
| Environment check (AWS) | `python scripts/check_env.py --strict` |
| MMPose cross-check (≤ 2 px) | `python scripts/crosscheck_mmpose.py --config CFG --checkpoint COCO.pth` |
| E0: generic COCO model | `python scripts/eval_generic_pose.py --config CFG --checkpoint COCO.pth --split data/manifests/pose_split_v1.json` |
| Tracker | `python scripts/run_tracker.py --siamfc-repo DIR --weights siamfc_alexnet_e50.pth --pilot-segments` (or `--skitb-root … --split-json … --partition val --fps N`) |
| Centered replay | `python scripts/make_replay.py --track-csv CLIP.csv --video CLIP.mp4` (τ/k grid on val: `--grid …`) |
| E1 training | `python scripts/train_pose.py --split data/manifests/pose_split_v1.json --hrnet-config CFG --checkpoint COCO.pth` (first `--overfit 32`, then `--profile-steps 200`) |
| E2–E4 training | the same, plus `--config configs/pose/e2.yaml --experiment E2` (likewise E3, E4); extra seeds: `--seed 231` |
| Augmentation preview (pilot images) | `python scripts/show_augmentations.py` → `$SKIER_RESULTS/pose/aug_preview/` |
| **Freeze the crop-error set** (after the split) | `python scripts/make_crop_error_set.py` → `data/manifests/crop_error_set_v1.json` |
| Evaluate a checkpoint (val) | `python scripts/eval_pose.py --checkpoint $SKIER_RESULTS/pose/E2/best.pt --hrnet-config CFG --split data/manifests/pose_split_v1.json --crop-error-set data/manifests/crop_error_set_v1.json` |
| Compare experiments | `python scripts/compare_experiments.py E0=…/E0/val/metrics.json E1=…/E1/eval/val/metrics.json E2=… E3=… E4=…` (table + D16 selection rule) |

`CFG` and `COCO.pth` come from `mim download mmpose --config td-hm_hrnet-w32_8xb64-210e_coco-256x192 --dest $SKIER_DATA/pretrained`.

## Evaluation protocol (summary)

The full definitions are in `docs/evaluation_protocol.md`.

- **Split:** whole source videos only. The plan's dry run gives train 65% / val 16% / test 19%. The 6 follow-cam videos form a separate evaluation set that is never trained on (D11). The split is frozen before any model is scored.
- **Pose:** PCK@0.1 (and @0.05 for skis), normalized by the diagonal of the reliable body + ski box. Body and ski are reported separately; a missing prediction counts as a miss; hits and totals are always reported.
- **Tracking:** mean IoU on visible frames (excluding the initialization frame), a success curve, and failure episodes.
- **Replay:** containment (ski endpoints separately), centering, crop acceleration and zoom, always reported together.
- **Usage:** all tuning happens on val. Test is evaluated once.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/skier/` | The package: `data/`, `pose/`, `tracking/`, `replay/`, `eval/`, plus `geometry`, `schema`, `splits`, `viz` |
| `scripts/` | Command-line entry points (thin wrappers) |
| `configs/` | `ski24_schema.json`, `pose/e1.yaml`–`e4.yaml`, `replay/default.yaml` |
| `data/manifests/` | `pose_inventory.csv`, `pilot_videos.txt`, and (after the freeze) `pose_split_v1.json` and `crop_error_set_v1.json` |
| `docs/` | `scope.md`, `decisions.md`, `data_audit_ski2dpose.md`, `data_audit_skitb.md`, `evaluation_protocol.md` |
| `tests/` | Unit tests on synthetic data (`tests/synth.py`) |
| `results/` | Small committed evidence; large outputs stay under `$SKIER_RESULTS` |
| `skier_implementation_plan.md`, `skier_step_1_guide.md`, `skier_step_1_code_spec.md`, `PROJECT_NOTES.md` | The plan, the setup guide, the Step 1 spec, and the running notes |

## References

- R. Bachmann, J. Spörri, P. Fua, H. Rhodin. *Motion capture from pan-tilt cameras with unknown orientation.* 3DV 2019 (Ski 2DPose).
- M. Dunnhofer, L. Sordi, N. Martinel, C. Micheloni. *Tracking skiers from the top to the bottom.* WACV 2024 (SkiTB).
- K. Sun, B. Xiao, D. Liu, J. Wang. *Deep high-resolution representation learning for human pose estimation.* CVPR 2019 (HRNet).
- L. Bertinetto et al. *Fully-convolutional siamese networks for object tracking.* ECCV Workshops 2016 (SiamFC).
- MMPose Contributors. *OpenMMLab pose estimation toolbox and benchmark*, 2020.
