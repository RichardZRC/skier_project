# Skier tracking and 2D body-and-ski landmark estimation

CS230 project, Fall 2026 · Ruichao Zhang (ruichaoz@stanford.edu)

Given a single-camera skiing video and a box around the skier in the first frame, the system:
- tracks the skier with a fixed, pretrained tracker: **SiamFC** (the main result) or **STARK-ST50** (compared with it in an extra experiment, D28; see [Trackers](#trackers-siamfc-and-stark));
- predicts **24 landmarks** (14 body, 8 ski, 2 pole) with a fine-tuned COCO HRNet-W32 on a crop around the tracker box;
- predicts again on a crop around those landmarks (**pose-guided crop refinement**, D26);
- produces a smooth, skier-centered 16:9 replay with the skeleton drawn on it.

The learning study asks whether targeted augmentation (crop jitter, motion blur, occlusion) improves ski-landmark accuracy and robustness to imperfect crops. Proposal: `skier_proposal.pdf`. Plan: `skier_implementation_plan.md`.

```mermaid
flowchart LR
    A["Video + first-frame box"] --> B["Fixed tracker: SiamFC (main) or STARK-ST50"]
    B --> C["Crop + fine-tuned 24-point HRNet"]
    C --> R["Refinement: re-crop around the landmarks, predict again"]
    B --> E["Smoothed 16:9 replay window"]
    R --> F["Landmarks drawn into the replay"]
    E --> F
```

## Status (2026-11-15)

**All steps are done, and the test set was evaluated once on 2026-11-12 (D27).** One extra tracker run on test followed on 2026-11-15 (STARK-ST50, D28), reported separately. The final report is due Fri 2026-11-20 (`final_report/`).

- **Results:** `results/step8/test/tables.md` (test), `results/step8/followcam/tables.md` (follow-cam set), `results/step8/val/tables.md` (val, which chose every setting).
  - E2 on test, reference crops: ski PCK@0.1 94.8%, body 95.9%.
  - Tracker crops cost 28.0 ski points; refinement recovers +9.3 [6.7, 12.1] of them, with no retraining.
  - SiamFC on 471 SkiTB test clips: mean IoU 0.404, toolkit F-score 0.406.
- **Extra experiment, STARK-ST50 in place of SiamFC (D28):** `results/step7b/test/compare.md`. SkiTB mean IoU 0.679 vs. 0.404; E2 ski PCK@0.1 under tracker crops 86.5% vs. 67.7%. SiamFC stays the main result.
- **Decisions:** D1–D28 in `docs/decisions.md`; running notes in `PROJECT_NOTES.md`.
- **Tests:** 344 pass on the GPU machine (a Windows laptop with WSL2, D17; `docs/laptop_setup.md`). In the light environment 15 skip (they need torch, mmpose, CUDA, the vendored STARK or SkiTB toolkit, or matplotlib).

## Setup

**Light environment (any machine; enough for every unit test):**

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # numpy<2, opencv-python, pillow, pyyaml, pytest
pytest -q                        # torch/stack/gpu tests skip without those packages
```

**GPU environment:** the pinned stack is torch 2.1.2 + cu121, mmcv 2.1.0, mmengine 0.10.x, mmpose 1.3.2, numpy < 2; the pins are listed in `requirements-pose.txt`.
- **Windows laptop (current, D17):** WSL2 Ubuntu 22.04, step by step in `docs/laptop_setup.md`.
- **AWS g5.xlarge (later):** install exactly as in `skier_step_1_guide.md` §1C.7.
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
| Tracker | `python scripts/run_tracker.py --siamfc-repo DIR --weights siamfc_alexnet_e50.pth --pilot-segments` (or `--skitb-root … --split-json … --partition val --fps N`); STARK: `--tracker stark --stark-repo DIR --weights STARKST_ep0050.pth.tar` |
| Compare two trackers | `python scripts/compare_trackers.py --a $SKIER_RESULTS/final/val --b $SKIER_RESULTS/final_stark/val --config-b configs/integration/stark.yaml --partition val --skitb-root $SKIER_DATA/skitb --skitb-split SPLIT.json` (`--config-a` defaults to SiamFC's `default.yaml`) |
| Centered replay | `python scripts/make_replay.py --track-csv CLIP.csv --video CLIP.mp4` (τ/k grid on val: `--grid …`) |
| E1 training | `python scripts/train_pose.py --split data/manifests/pose_split_v1.json --hrnet-config CFG --checkpoint COCO.pth` (first `--overfit 32`, then `--profile-steps 200`) |
| E2–E4 training | the same, plus `--config configs/pose/e2.yaml --experiment E2` (likewise E3, E4); extra seeds: `--seed 231` |
| Augmentation preview (pilot images) | `python scripts/show_augmentations.py` → `$SKIER_RESULTS/pose/aug_preview/` |
| **Freeze the crop-error set** (after the split) | `python scripts/make_crop_error_set.py` → `data/manifests/crop_error_set_v1.json` |
| Evaluate a checkpoint (val) | `python scripts/eval_pose.py --checkpoint $SKIER_RESULTS/pose/E2/best.pt --hrnet-config CFG --split data/manifests/pose_split_v1.json --crop-error-set data/manifests/crop_error_set_v1.json` |
| Compare experiments | `python scripts/compare_experiments.py E0=…/E0/val/metrics.json E1=…/E1/eval/val/metrics.json E2=… E3=… E4=…` (table + D16 selection rule) |

`CFG` and `COCO.pth` come from `mim download mmpose --config td-hm_hrnet-w32_8xb64-210e_coco-256x192 --dest $SKIER_DATA/pretrained`.

## Reproducing the reported results

On the GPU machine, inside Ubuntu, with the data in `$SKIER_DATA` and the vendored SiamFC fork and weights in `$SKIER_DATA/vendor/siamfc_pytorch` (`docs/laptop_setup.md`). Each training run takes about 40 minutes on an RTX 4070.

```bash
SPLIT=data/manifests/pose_split_v1.json
CFG=$SKIER_DATA/pretrained/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py
COCO=$SKIER_DATA/pretrained/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth
SIAM="--siamfc-repo $SKIER_DATA/vendor/siamfc_pytorch --weights $SKIER_DATA/vendor/siamfc_pytorch/pretrained/siamfc_alexnet_e50.pth"

# 1. Augmentation study (D21): E1-E4, seeds 230 and 231, 150 fine-tune epochs (set in configs/pose/e*.yaml)
for e in 1 2 3 4; do for s in 230 231; do
  python scripts/train_pose.py --split $SPLIT --hrnet-config $CFG --checkpoint $COCO \
    --config configs/pose/e$e.yaml --experiment E$e --seed $s --out $SKIER_RESULTS/pose/d21/E$e-s$s
done; done
# then evaluate each on val (scripts/eval_pose.py) and apply the D16 rule (scripts/compare_experiments.py): picks E2

# 2. Val-only choices (Step 7; each by a rule written in advance, D24 and D26)
python scripts/check_tracker_confidence.py --track-dir TRACKS --skitb-root $SKIER_DATA/skitb \
  --split-json $SKIER_DATA/skitb/AL/AL_train_val_test_athlete_60-40.json      # confidence signal and threshold
python scripts/eval_propagation.py --split $SPLIT --partition val --usable USABLE.json --track-dir TRACKS2D \
  --checkpoint E2-s230=$SKIER_RESULTS/pose/d21/E2-s230/best.pt ... --hrnet-config $CFG --refine-grid
                                                    # k_pose grid and the 12-setting refinement grid

# 3. The frozen evaluation: the identical chain on val, then test once (refuses a second run)
bash results/step8/run_final.sh val
bash results/step8/run_final.sh test

# 4. One clip end to end, with refinement on (configs/integration/default.yaml)
python scripts/process_clip.py --skitb-clip AL0027/3 --skitb-root $SKIER_DATA/skitb \
  --split-json $SKIER_DATA/skitb/AL/AL_train_val_test_athlete_60-40.json --partition val $SIAM \
  --checkpoint $SKIER_RESULTS/pose/d21/E2-s230/best.pt --hrnet-config $CFG --out $SKIER_RESULTS/final/demos/AL0027__3
```

- `run_final.sh` writes `freeze.json` (hashes of the splits, configs and checkpoints) before any output, and checks that the tree is clean and equal to `origin/main`.
- Report figures: `milestone/figures/*.py` and `final_report/figures/*.py`. The qualitative figure shows dataset frames, so it is not committed.
- The STARK comparison is reproduced separately: see [Trackers](#trackers-siamfc-and-stark).

## Trackers: SiamFC and STARK

Both trackers are in the code. Every tracking tool (`run_tracker.py`, `process_clip.py`, `smoke_tracker.py`) takes `--tracker {siamfc,stark}`, and **SiamFC is the default**, so every command above runs SiamFC unchanged.

| | SiamFC (main result) | STARK-ST50 (extra experiment, D28) |
| --- | --- | --- |
| Method | Siamese AlexNet, cross-correlation with the first-frame template | ResNet-50 + transformer, direct box prediction, learned confidence score, template update every 200 frames |
| Code and weights | fork of `siamfc-pytorch` @ `a87f140`, `siamfc_alexnet_e50.pth` | `researchmm/Stark` @ `162e1d3`, `STARKST_ep0050.pth.tar` (`docs/laptop_setup.md` §3.4) |
| Adapter | `src/skier/tracking/siamfc_adapter.py` | `src/skier/tracking/stark_adapter.py` (STARK unedited under torch 2.1; no extra packages) |
| Integration config | `configs/integration/default.yaml` | `configs/integration/stark.yaml` |
| Final runs | `results/step8/run_final.sh` → `$SKIER_RESULTS/final/<part>/` | `results/step7b/run_stark.sh` → `$SKIER_RESULTS/final_stark/<part>/` |
| Evidence | `results/step8/` | `results/step7b/` |

- **Fixed, not tuned:** both trackers run off the shelf with their own published settings. With STARK, only the tracker changes; the pose models, crop factor, refinement and replay settings are SiamFC's.
- **Confidence threshold per tracker:** STARK's score is a probability in [0, 1], SiamFC's is about 1e-5, so each integration config has its own threshold, chosen on val by the same rule (D24). Every track folder records its tracker in `tracker.json`, and the tools refuse a track folder whose tracker doesn't match the config.
- **Order of runs:** the STARK experiment came after SiamFC's single test run. Its settings and comparisons were fixed on val, then it ran once on test (2026-11-15), whatever the val result. It is reported as an extra experiment and never replaces SiamFC's numbers.

Reproducing the comparison (after `run_final.sh` has produced SiamFC's run of the same set):

```bash
STARK=$SKIER_DATA/vendor/Stark
W=$STARK/checkpoints/train/stark_st2/baseline/STARKST_ep0050.pth.tar
python scripts/run_tracker.py --tracker stark --stark-repo $STARK --weights $W \
  --skitb-root $SKIER_DATA/skitb --split-json $SKIER_DATA/skitb/AL/AL_train_val_test_athlete_60-40.json \
  --partition val --out $SKIER_RESULTS/tracking/skitb-val-stark
python scripts/check_tracker_confidence.py --track-dir $SKIER_RESULTS/tracking/skitb-val-stark \
  --skitb-root $SKIER_DATA/skitb --split-json $SKIER_DATA/skitb/AL/AL_train_val_test_athlete_60-40.json
                                    # STARK's confidence signal and threshold → configs/integration/stark.yaml
bash results/step7b/run_stark.sh val     # the val comparison (compare/compare.md)
bash results/step7b/run_stark.sh test    # once; refuses a second run
```

`run_stark.sh` checks the STARK commit and the weights' SHA-256, writes `freeze.json` before any output, refuses test unless the tree is clean and equal to its `origin` branch, and checks that SiamFC's `final/<part>/` is unchanged at the end.

Test results (`results/step7b/test/compare.md`; paired 95% intervals, D25):

| | SiamFC | STARK-ST50 | Difference [95% CI] |
| --- | --- | --- | --- |
| SkiTB mean IoU (471 clips) | 0.404 | 0.679 | +0.276 [0.251, 0.301] |
| SkiTB toolkit F-score | 0.406 | 0.677 | +0.270 [0.246, 0.294] |
| E2 body PCK@0.1, tracker crop | 78.9% | 96.5% | +17.6 [9.1, 27.0] |
| E2 ski PCK@0.1, tracker crop | 67.7% | 86.5% | +18.8 [11.1, 27.8] |
| E2 ski PCK@0.1, refined crop | 77.0% | 92.1% | +15.1 [7.6, 24.1] |
| Replay: ski ends in view | 93.7% | 99.8% | +6.2 [0.6, 13.2] |

## Evaluation protocol (summary)

The full definitions are in `docs/evaluation_protocol.md`.

- **Split:** whole source videos only. The plan's dry run gives train 65% / val 16% / test 19%. The 6 follow-cam videos form a separate evaluation set that is never trained on (D11). The split is frozen before any model is scored.
- **Pose:** PCK@0.1 (and @0.05 for skis), normalized by the diagonal of the reliable body + ski box. Body and ski are reported separately; a missing prediction counts as a miss; hits and totals are always reported.
- **Tracking:** mean IoU on visible frames (excluding the initialization frame), a success curve, and failure episodes.
- **Replay:** containment (ski endpoints separately), centering, crop acceleration and zoom, always reported together.
- **Usage:** all tuning happens on val. Test is evaluated once (done 2026-11-12, D27). The STARK extra experiment ran once on test afterwards (D28) and is reported separately.
- **Intervals:** a cluster bootstrap over segments within videos (Ski 2DPose) or sequences (SkiTB), 10,000 resamples (D25); the follow-cam set resamples its 6 videos, each one segment (D27).

## Repository layout

| Path | Contents |
| --- | --- |
| `src/skier/` | The package: `data/`, `pose/`, `tracking/`, `replay/`, `integration/` (confidence, propagation, refinement, one-clip pipeline), `eval/` (metrics, bootstrap, SkiTB toolkit port), `report/` (tables, figures), plus `geometry`, `schema`, `splits`, `viz` |
| `scripts/` | Command-line entry points (thin wrappers) |
| `configs/` | `ski24_schema.json`, `pose/e1.yaml`–`e4.yaml`, `replay/default.yaml`, `integration/default.yaml` (SiamFC), `integration/stark.yaml` (STARK) |
| `data/manifests/` | `pose_inventory.csv`, `pilot_videos.txt`, the frozen `pose_split_v1.json` and `crop_error_set_v1.json`, `skitb_fps_v1.csv` |
| `docs/` | `scope.md`, `decisions.md`, `data_audit_ski2dpose.md`, `data_audit_skitb.md`, `evaluation_protocol.md`, `laptop_setup.md` |
| `tests/` | Unit tests on synthetic data (`tests/synth.py`) |
| `results/` | Small committed evidence per step (`results/step8/{val,test,followcam}/` for the final numbers, `results/step7b/{val,test}/` for the STARK comparison); large outputs stay under `$SKIER_RESULTS` |
| `milestone/`, `final_report/` | The CS230 milestone and final report (LaTeX, figures and the scripts that draw them) |
| `skier_implementation_plan.md`, `skier_step_1_guide.md`, `skier_step_1_code_spec.md`, `PROJECT_NOTES.md` | The plan, the setup guide, the Step 1 spec, and the running notes |

## References

- R. Bachmann, J. Spörri, P. Fua, H. Rhodin. *Motion capture from pan-tilt cameras with unknown orientation.* 3DV 2019 (Ski 2DPose).
- M. Dunnhofer, L. Sordi, N. Martinel, C. Micheloni. *Tracking skiers from the top to the bottom.* WACV 2024 (SkiTB).
- K. Sun, B. Xiao, D. Liu, J. Wang. *Deep high-resolution representation learning for human pose estimation.* CVPR 2019 (HRNet).
- L. Bertinetto et al. *Fully-convolutional siamese networks for object tracking.* ECCV Workshops 2016 (SiamFC).
- B. Yan, H. Peng, J. Fu, D. Wang, H. Lu. *Learning spatio-temporal transformer for visual tracking.* ICCV 2021 (STARK).
- MMPose Contributors. *OpenMMLab pose estimation toolbox and benchmark*, 2020.
