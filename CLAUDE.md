# CLAUDE.md

CS230 project (Ruichao Zhang): skier tracking, a centered replay, and 24 body/ski/pole landmarks. The learning component is fine-tuning COCO HRNet-W32 plus an augmentation study (decision D1).

**Deadline: Fri 2026-11-20** (D14; plan §4). If the schedule slips, cut stretch goals first, then extra seeds, then E2/E3.

**Read first:** `PROJECT_NOTES.md` (start with its **Handoff** section: machine state, what's waiting, AWS lessons; then status, next steps, working rules) → `skier_implementation_plan.md` (the plan; each step has a status note) → `docs/decisions.md` (D1–D28; append only).

## Working rules (Ruichao's instructions; follow them exactly)

- **Plan mode before every plan step.** Enter plan mode, write the plan, and get it approved before coding Step N. Small fixes inside an approved step don't need a new plan.
- **Install nothing on this Mac (D13).** No torch, no mmpose, no new packages. Only the light `.venv` exists here. Torch- and mmpose-dependent tests skip here and run on the GPU machine (the laptop for now, D17).
- **Commit and push only when asked.** Work directly on `main`; no branches.
- **Never widen the sandbox.** pypi.org and the EPFL dataset server are blocked from this session. Ruichao downloads in their own terminal. Don't edit the ACC allowlists or sandbox settings.
- **Data discipline:**
  - Never commit dataset images or labels.
  - Before the split is frozen, look only at **pilot-video** images (`data/manifests/pilot_videos.txt`).
  - Never browse test images. Tune on **val** only. Evaluate **test once** (Step 8).
  - The split manifest is frozen: `make_pose_split.py` refuses to overwrite it without `--force` and a recorded reason.
- **Record as you go.** Each new decision goes into `docs/decisions.md`. After each step, update the status in `PROJECT_NOTES.md` and the step's status note in the plan. Knowledge a later session needs goes into the repo (the Handoff section), because Claude's own notes stay on the machine where they were written.
- **Working style:**
  - Give step-by-step walkthroughs for console and infrastructure tasks. Ruichao runs commands on the other machines and pastes the output.
  - Run a cheap trial before any compute-heavy work.
  - Keep options open: don't drop plans, move doable work to whatever machine is available.
  - "Don't do anything yet, just tell me" means answer only.

## Environment

- Run the tests with `.venv/bin/python -m pytest -q`. Currently **344 tests**; all must pass on the GPU machine. The Mac skips 15: 7 `torch`, 4 `stack`, the real-STARK test (it needs CUDA and the vendored STARK), the SkiTB-toolkit cross-check (it needs the vendored toolkit) and two figure tests (they need matplotlib).
  - Markers: `torch` needs PyTorch; `stack` needs the pinned torch, mmcv and mmpose; `gpu` needs CUDA.
- **Paths:** code reads `SKIER_DATA` (default `~/skier_data`) and `SKIER_RESULTS` (default `~/skier_results`) through `skier.paths`. Never hard-code absolute paths.
- **Data on this Mac:**
  - Ski 2DPose (audited 2026-10-17, 0 problems): `~/skier_data/ski2dpose/{archives,labels,images/Images_jpg,videos/Videos}`. Checksums: `docs/data_audit_ski2dpose.md`.
  - SkiTB (audited 2026-10-18, 0 format problems): `~/skier_data/skitb/{AL,FS,JP}`; only `AL/` is used. Digests and findings: `docs/data_audit_skitb.md`.
- **Which machine is this session on?**
  - **Mac** (macOS, repo in `~/Desktop/skier_project`): coding, the light tests, reviews, the split freeze. No installs (D13); the sandbox rules above apply.
  - **Laptop** (WSL2 Ubuntu, repo in `~/skier_project`): GPU runs. The pinned stack is installed there as in `docs/laptop_setup.md`; ask before installing anything else. Data lives only in `~/skier_data` inside WSL (SkiTB: just the val/test subset; vendor code in `~/skier_data/vendor`). Never commit images, labels, checkpoints or `.npz` files.
  - **Laptop, Windows side** (PowerShell, repo in `OneDrive\Desktop\skier_project`): the usual setup since 2026-10-24.
    - **Running commands:** Claude edits here, and runs commands in Ubuntu itself via `wsl -d Ubuntu-22.04 -- bash <script>`. Ruichao asked for this, so there are no walkthroughs here. Put multi-line commands in a script file in the scratchpad, because PowerShell mangles `$` and quotes.
    - **Committing:** commits happen in this clone. Copy edited files into the WSL clone with `sed -i 's/\r$//'`, and `git checkout -- <files>` there before `git pull`.
    - **Long runs** go in tmux inside Ubuntu.
- **GPU machines (D17):** Ruichao's Windows laptop (RTX 4070, WSL2 Ubuntu 22.04) for now, set up as in `docs/laptop_setup.md`. AWS g5.xlarge (guide 1C.7) is kept for later; its account verification is still pending. Ruichao runs the commands there and pastes the output; code moves through GitHub.
- **Pinned GPU stack** (GPU machines only): Python 3.10, torch 2.1.2 + cu121, mmcv 2.1.0, mmengine 0.10.x, mmpose 1.3.2, mmdet 3.2.0, numpy < 2, opencv-python < 4.12 (D18).
- The repo sits in `~/Desktop`, which iCloud syncs. Data and results live outside it, as they should.

## Code conventions

- **Package and scripts:** logic lives in `src/skier/`. `scripts/*.py` are thin wrappers around `main(argv)`. Every tool takes `--out` and defaults under `$SKIER_RESULTS`.
- **Lazy heavy imports:** torch, mmpose and siamfc are imported only inside the functions that need them. `tests/test_cli.py` checks that every script's `--help` works without them, so register new scripts there.
- **Torch-free logic:** keep logic testable with numpy and OpenCV. Examples: the model enters inference as a plain `heatmap_fn(np) -> np`, and the loss has a numpy reference, `masked_mse_np`.
- **Coordinates:**
  - original-image pixels, `x = x_norm × W`, using each image's own size;
  - OpenCV pixel centers; boxes are `xyxy`;
  - `sample_id = "{video}/{segment}/{image}"`, and `safe_name()` makes it filename-safe.
- **MMPose conventions (D12):**
  - crop: pad 1.25 once → fit to 192:256 → warp → RGB → mean/std;
  - heatmap coordinate = input / 4, with no half-pixel shift;
  - targets are sub-pixel Gaussians; the decoder is a verbatim copy of MMPose's MSRA decoder (edge conditions included);
  - flip maps `x → 191 − x` and swaps left/right labels.
- **Unknown facts:** anything only the real data can answer sits in a `FILL IN` block (code spec §13). Never guess it; fail loudly.
- **Evidence in git:** small text evidence (JSON/CSV/TXT) goes to `results/stepN/`. Large outputs stay under `$SKIER_RESULTS`.
- **Tests:** synthetic inputs come from `tests/synth.py`; tests always write under `tmp_path`. For every new module, run a mutation check: plant bugs one at a time and confirm a test fails each time.

## Gotchas already hit

- **Mutation checks and stale bytecode:** use `PYTHONDONTWRITEBYTECODE=1` and clear `__pycache__`. Python's bytecode cache checks only modification time and size, so a same-size mutant restored within the same second leaves stale `.pyc` files.
- **Flip test in MMPose:** the COCO config enables it by default. The cross-check turns it off (`model.test_cfg.flip_test=False`) because our pipeline never flips.
- **Decoder edges:** the decoder skips the quarter-pixel step in the outer heatmap pixels. Error bounds are ~1.4 input px inside and ~2.25 px at the edges.
- **Acceleration tests:** evenly spaced timestamps can't catch a wrong acceleration formula. Keep the uneven-timestep test.
- **Checkpoints:** load with `torch.load(..., weights_only=False)`, because our checkpoints carry config and metadata. MMPose's COCO checkpoint also carries `data_preprocessor.mean/std`; `build_hrnet` checks them against `preprocess.MEAN/STD` and leaves them out.
- **Profiling and overfit runs:** `--profile-steps` stops inside the frozen warm-up epochs. Profile the fine-tune phase with `results/step5/profile_finetune_config.yaml` instead. `--overfit 32` needs `--epochs 300` before ski PCK reaches ~100%.
- **Two clones on the laptop:** the Windows copy (`OneDrive\Desktop`) checks out with CRLF line endings. Files copied from it into the WSL clone need `sed -i 's/\r$//'`, or `git diff` shows every line changed.
- **Training schedule (D20, D21):** at 50 fine-tune epochs the ski tails (and sometimes tips) are learned only in some runs, depending on the seed. Use 150 epochs and 2 seeds for comparisons. Check convergence per joint on training images (`results/step6/trial_eval.py`), not only val ski PCK.
- **SiamFC on broadcast video:** after a few seconds it can drift to clutter and lock onto static on-screen graphics such as the logo (tracker smoke test, 2026-10-26). It also drifts to spectators, and its box can grow 8–10× too large (Step 7 failure review).
- **SiamFC's score** is the peak of a sum-normalized response map, about 1e-5. Track CSVs keep 7 significant digits; 5 decimals collapsed it to about 7 values. The gating threshold (D24) is on this raw scale.
- **`wsl -- bash -c "..."`** expands `$VAR` before bash sees it, whether called from PowerShell or Git Bash. Put anything with variables in a script file.
- **The SkiTB toolkit doesn't run as vendored:** `np.loadtxt(delimiter='\n')` fails on numpy 1.26, and SC mode lists `ST/` instead of `SC/`. Use `eval/skitb_official.py`; `results/step8/toolkit_crosscheck.py` shows the port matches `report()`.
- **Laptop interruptions:** Windows sleep (Modern Standby), Windows Update restarts and closing the last `wsl` connection have each stopped a run. Every epoch is checkpointed; resume with `--resume <run>/last.pt` and the same arguments.
- **STARK (D28):** its scores are a sigmoid in [0, 1], SiamFC's about 1e-5, so each tracker has its own integration config (`stark.yaml` has `tracker: stark`); the tools refuse a track folder whose `tracker.json` doesn't match. STARK's code needs six torch-2 workarounds, all in `stark_adapter.py`; never edit the clone. Both trackers stay in the code (`--tracker`, SiamFC the default); the STARK work was done on branch `stark` and fast-forwarded into `main` on 2026-11-15 (Ruichao: keep both, for the comparison).
- **Minimum crop:** replay crops are at least `max(240 px, frame_h / 3)`, so design controller tests around that.
- **SkiTB:**
  - `frames.txt` holds bare frame indices, and `frame_file` maps them (`197` → `00197.jpg`).
  - The frame rate isn't in the files or the toolkit, and differs by sequence. Use the frozen per-sequence table `data/manifests/skitb_fps_v1.csv` (D22: 25 or 30 fps), which `run_tracker.py` and `make_replay.py` load by default.
  - Copies made from the Mac onto a non-Mac drive bring macOS `._*` sidecar files; remove them before auditing.
  - Frame sizes vary, mostly 1280×720.
  - The "athlete" split shares 5 athletes between train and test.

## Module map

| Area | Modules (`src/skier/…`) |
| --- | --- |
| Data | `data/ski2dpose_raw.py` (raw labels, path template, info), `data/ski2dpose.py` + `data/records.py` (`PoseRecord`), `data/audit.py`, `data/overlays.py`, `data/alignment.py`, `data/skitb.py` (layout confirmed 2026-10-18), `splits.py` |
| Geometry and drawing | `geometry.py`, `viz.py`, `schema.py` (+ `configs/ski24_schema.json`) |
| Pose | `pose/preprocess.py`, `pose/heatmap.py`, `pose/infer.py`, `pose/model.py`, `pose/coco_map.py`, `pose/eval_generic.py` (E0), `pose/crosscheck.py`, `pose/augment.py`, `pose/dataset.py`, `pose/loss.py`, `pose/train.py`, `pose/crop_error.py`, `pose/evaluate.py`, `pose/compare.py`, `pose/aug_preview.py` |
| Tracking and replay | `tracking/siamfc_adapter.py`, `tracking/stark_adapter.py` (D28), `tracking/trackers.py` (tracker choice, `tracker.json`, the guard), `tracking/frames.py`, `tracking/run.py`, `replay/controller.py`, `replay/render.py`, `replay/run.py` |
| Integration (Step 7) | `integration/config.py` (+ `configs/integration/default.yaml`), `integration/confidence.py`, `integration/propagation.py`, `integration/refine.py` (D26 crop refinement), `integration/process.py` |
| Metrics | `eval/pose_metrics.py`, `eval/track_metrics.py`, `eval/replay_metrics.py`, `eval/bootstrap.py` (D25), `eval/skitb_official.py` (the toolkit's F/Pr/Re/GSR, ported) |
| Report (Step 8) | `report/tables.py` (`make_tables.py`), `report/figures.py` (`make_figures.py`), `report/compare_trackers.py` (`compare_trackers.py`, D28); the frozen run is `results/step8/run_final.sh`, the STARK extra `results/step7b/run_stark.sh` |
| Tools | `envcheck.py`, `io_utils.py`, `paths.py` |
