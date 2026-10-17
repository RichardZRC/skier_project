# Step 1 guide: setup and feasibility (v2)

**Owner:** Ruichao Zhang — ruichaoz@stanford.edu
**Guide version:** v2, October 17, 2026 (replaces v1; v1 is in git history)
**Time budget:** about 12 active hours (Week 1, starting 2026-10-19)
**Full plan:** [Six-week implementation plan](skier_implementation_plan.md)

Tags: `[CODE]` pure coding (no internet, data or GPU) · `[NET]` internet · `[DATA]` real dataset · `[GPU]` CUDA GPU · `[MANUAL]` you.

## What Step 1 must establish

1. The images, labels and videos are on the AWS instance, with provenance recorded.
2. Our code reads the labels correctly: joint order, left/right, coordinates, visibility.
3. Labeled images can be matched to the right video frames.
4. The pinned environment runs a real pretrained pose model on the AWS GPU, and SiamFC runs on a short clip.

The output is a small evidence package and a decision: **ready**, **ready with a limitation**, or **blocked**. No training happens in this step.

## Session plan

| Task | Tags | Hours | Evidence |
| --- | --- | ---: | --- |
| 1A. Repo scaffold | `[CODE]` | 0.5 | Package skeleton, tests run |
| 1B. Offline coding block: all Step 1 scripts | `[CODE]` | 3 | Scripts + passing synthetic tests |
| 1C. AWS account, instance and environment | `[MANUAL][NET]` | 2 | Running instance, `env.json` |
| 1D. Download data and record provenance | `[NET][DATA]` | 1 | Download ledger, checksums |
| 1E. Label audit and inventory | `[DATA]` | 1 | `label_summary.json`, inventory CSV |
| 1F. Ground-truth overlays + your review | `[DATA][MANUAL]` | 1.5 | Overlay gallery, audit notes |
| 1G. Video-frame alignment | `[DATA][MANUAL]` | 1 | `alignment.csv`, GT clip |
| 1H. Pose and tracker smoke tests | `[DATA][GPU]` | 1.5 | Predictions, timing |
| 1I. Readiness decision | `[MANUAL]` | 0.5 | `docs/step1_readiness.md` |
| **Total** | | **12** | |

**Do first, today:** the AWS budget alert and GPU quota request in 1C, which can take a day or more to approve. SkiTB access was requested on 2026-10-17; follow up if there's no reply by 2026-10-24.

**Order:** start 1C's quota request → do 1A + 1B offline while waiting → finish 1C → 1D–1I on the instance.

---

## 1A. Repo scaffold `[CODE]`

Work in the existing repo (`git@github.com:RichardZRC/skier_project.git`); don't create a new folder. Clone it outside any cloud-sync folder.

Create:

```text
docs/  configs/  configs/pose/  configs/replay/  data/manifests/
src/skier/  src/skier/data/  src/skier/pose/  src/skier/tracking/
src/skier/replay/  src/skier/eval/  scripts/  tests/  tests/fixtures/  results/
```

- `pyproject.toml`: package `skier` under `src/`, installable with `pip install -e .`; `pytest` config pointing at `tests/`.
- `requirements-pose.txt`: the pinned stack from section 1C (comments mark which lines need special index URLs).
- `src/skier/paths.py`: `data_root()` reads `SKIER_DATA` (default `~/skier_data`), `results_root()` reads `SKIER_RESULTS` (default `~/skier_results`).
- `.gitignore`:

  ```gitignore
  # large or machine-specific
  data/raw/
  data/processed/
  artifacts/
  *.pth
  *.pt
  *.mp4
  *.avi
  *.zip
  __pycache__/
  .venv/
  .pytest_cache/
  CLAUDE.local.md
  ```

- `docs/scope.md`: core, stretch order, research question, exclusions (copy from section 1 of the main plan).
- `docs/decisions.md`: one entry per decision with date, reason and effect on evaluation. Seed it with D1–D9 from the main plan.

**Done when:** `pip install -e .` and `pytest` run (zero tests is fine at this point).

---

## 1B. Offline coding block `[CODE]`

Write all of these without data or internet. Each has a synthetic test. Unknowns that only the real files can answer are isolated in one clearly marked place, filled in during 1E.

### 1B.1 Landmark schema — `configs/ski24_schema.json` + `src/skier/schema.py`

Zero-based order from the official dataset page (verified 2026-10-17):

| Index | Name | Group | Flip partner |
| ---: | --- | --- | ---: |
| 0 | head | body | 0 |
| 1 | neck | body | 1 |
| 2 | shoulder_right | body | 6 |
| 3 | elbow_right | body | 7 |
| 4 | hand_right | body | 8 |
| 5 | pole_basket_right | pole | 9 |
| 6 | shoulder_left | body | 2 |
| 7 | elbow_left | body | 3 |
| 8 | hand_left | body | 4 |
| 9 | pole_basket_left | pole | 5 |
| 10 | hip_right | body | 13 |
| 11 | knee_right | body | 14 |
| 12 | ankle_right | body | 15 |
| 13 | hip_left | body | 10 |
| 14 | knee_left | body | 11 |
| 15 | ankle_left | body | 12 |
| 16 | ski_tip_right | ski | 20 |
| 17 | toes_right | ski | 21 |
| 18 | heel_right | ski | 22 |
| 19 | ski_tail_right | ski | 23 |
| 20 | ski_tip_left | ski | 16 |
| 21 | toes_left | ski | 17 |
| 22 | heel_left | ski | 18 |
| 23 | ski_tail_left | ski | 19 |

JSON fields:
- `names`: the 24 names.
- `groups`:
  - `body`: [0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15]
  - `ski`: [16, 17, 18, 19, 20, 21, 22, 23]
  - `pole`: [5, 9]
- `flip_pairs`: [[2,6],[3,7],[4,8],[5,9],[10,13],[11,14],[12,15],[16,20],[17,21],[18,22],[19,23]]
- `side`: `right` / `left` / `center` per index.
- `edges`, each tagged `body` / `ski` / `pole`:
  - Body: 0–1, 1–2, 1–6, 2–3, 3–4, 6–7, 7–8, 2–10, 6–13, 10–13, 10–11, 11–12, 13–14, 14–15.
  - Ski: 16–17, 17–18, 18–19, 20–21, 21–22, 22–23.
  - Pole: 4–5, 8–9.
- `coco_shared`: COCO → Ski pairs `{5:6, 6:2, 7:7, 8:3, 11:13, 12:10, 13:14, 14:11, 15:15, 16:12}`.
- `coco_approx`: `{9:8, 10:4}`, wrist → hand, reported separately.

"Right" means the skier's own right, not the image's right side.

Tests:
- 24 unique names.
- Groups partition 0–23 with sizes 14 / 8 / 2.
- Flip mapping is symmetric, and flipping twice gives back the original order.
- Every flip pair maps right to left.
- Edges reference valid indices.

### 1B.2 Label audit and inventory — `scripts/audit_pose_labels.py`

```text
python scripts/audit_pose_labels.py LABELS_JSON --info INFO_JSON
       [--images-root DIR --path-template TEMPLATE] --out DIR
```

- **Pass 1, structure only.** Walk `video → segment → image`. Count videos, segments and images; images per video; visibility values seen; reliable count per joint. Flag:
  - annotation shape ≠ (24, 3);
  - non-finite values;
  - coordinates outside [0, 1];
  - visibility values other than 0 or 1;
  - missing `frame_idx`;
  - repeated `frame_idx` within a segment.

  Never clamp or fix anything; only report.
- **Pass 2, with images.** Resolve each record through `--path-template` (for example `{video_id}/{segment_id}/{image_id}.jpg`; the real pattern is filled in during 1E). Read each image's real width and height (header only). Flag labels without images, images without labels, and duplicate sample IDs.
- **Outputs:**
  - `label_summary.json`;
  - `problems.csv` with `sample_id, problem, detail`;
  - `data/manifests/pose_inventory.csv` with `sample_id, video_id, segment_id, image_id, frame_idx, image_path, width, height, n_reliable_body, n_reliable_ski, n_reliable_pole, discipline, view_angle`. `discipline` and `view_angle` come from the info JSON; leave them blank if the key names aren't known yet.
- Tests: a fixture JSON with 2 videos × 2 segments × 3 images, where one record has shape (23, 3), one has x = 1.2, one is missing `frame_idx`, and one image file is missing (blank PNGs generated in a temp dir). Each problem must be reported exactly once.

### 1B.3 Ground-truth overlays — `scripts/draw_gt_overlays.py`

```text
python scripts/draw_gt_overlays.py --inventory CSV --labels LABELS_JSON
       --videos-file data/manifests/pilot_videos.txt --n 30 --seed 230 --out DIR
```

- Picks `n` images from the pilot videos only, spread across segments and view angles.
- Drawing rules:
  - Skier's **right** side in orange, **left** side in cyan, head/neck in white.
  - **Filled** circle = reliable (`visible = 1`); **hollow** circle = occluded guess (`visible = 0`).
  - Each point labeled with its index number.
  - Edges: body thin, skis thick, poles dotted. An edge with an unreliable endpoint is drawn dashed.
  - Dashed yellow rectangle = reference box (reliable body + ski points, no poles).
  - Header text: sample ID, image size, discipline, view.
- Outputs: one full-resolution PNG per image, `contact_sheet.png` (6 × 5 thumbnails), and `overlay_ids.txt`.
- Tests: on a synthetic 200 × 100 blank image with three points, a reliable point's center pixel has the marker color and a hollow point's center pixel stays the background color. The coordinate conversion uses the image's real size.

**Pilot videos:** pick 2–3 videos with large, clearly visible skiers and different race types, and avoid the follow-cam videos. Save their IDs in `data/manifests/pilot_videos.txt`. They will be forced into **training** in Step 2 (`forced_train`), so looking at them closely never contaminates val or test.

### 1B.4 Video-frame alignment — `scripts/check_alignment.py`

```text
python scripts/check_alignment.py --inventory CSV --video FILE --video-id ID --segment-id SEG
       --window 5 --out DIR
```

- Decodes the video **sequentially** (no random seeking). For each labeled image in the segment, compares the image with frames `frame_idx − window … frame_idx + window`: convert both to grayscale at 160 × 90 and compute normalized cross-correlation.
- Reports the best offset and the gap between the best and second-best score. A small gap means the match is ambiguous.
- Outputs:
  - `alignment.csv` with `video_id, segment_id, image_id, label_frame_idx, best_decode_idx, offset, best_score, margin, verified`;
  - `gt_clip.mp4`, with labels drawn **only** on labeled frames.
- Whether the video archive holds one clip per segment or whole source videos is unknown until 1E. Keep the "which file, which start frame" logic in one function, `locate_segment(...)`. If they are whole videos, use the info JSON's split time to find the start.
- Tests: generate a 60-frame synthetic video of a moving square (MJPG `.avi`), save frames 10, 30 and 50 as images; the script must return offset 0. With labels deliberately shifted by +2, it must return −2.

### 1B.5 Environment check — `scripts/check_env.py`

- Prints and saves `artifacts/step1/env.json` with:
  - OS, Python version;
  - torch, torchvision, mmcv, mmengine, mmpose, numpy and opencv versions;
  - `torch.version.cuda`, `torch.cuda.is_available()`, GPU name and VRAM.
- Asserts `numpy < 2` and that `import mmcv.ops` works (proves the compiled extensions match).
- Runs one 3 × 3 convolution on the GPU and reports success.
- Test: runs on CPU without a GPU and reports `cuda_available: false` without crashing.

### 1B.6 Pose smoke test — `scripts/smoke_pose.py`

```text
python scripts/smoke_pose.py --config CFG --checkpoint PTH --image IMG --box x1 y1 x2 y2 --out DIR
```

- Uses MMPose's own `init_model` + `inference_topdown(model, img, bboxes, bbox_format="xyxy")` with the skier's **tight** box. MMPose adds its own 1.25 padding here; this is only a test of the install.
- Asserts 17 finite keypoints and scores.
- Saves `generic_pose_prediction.npz` and an overlay drawn with the **COCO 17-point** skeleton (never the 24-point one).
- Times 20 runs after 5 warm-up runs, calling `torch.cuda.synchronize()` around the timing.
- Offline test: none needed beyond argument parsing; this script exists to be run.

### 1B.7 Tracker smoke test — `scripts/smoke_tracker.py`

```text
python scripts/smoke_tracker.py --siamfc-repo DIR --weights PTH --video FILE
       --init-frame N --init-box x1 y1 x2 y2 --max-frames 300 --out DIR
```

- Minimal direct use of the `siamfc-pytorch` tracker. It is replaced by the proper adapter in Step 3.
- Initializes once and never resets.
- Writes `boxes.csv` (`frame_index, x1, y1, x2, y2`) and `track.mp4` with boxes drawn.
- Offline test: the box-drawing and CSV functions on a synthetic video with a fake tracker.

**Done when:** `pytest` passes for 1B.1–1B.4 and 1B.7, and every script shows `--help`.

---

## 1C. AWS setup `[MANUAL][NET]`

You don't need to understand all of AWS. You need one GPU machine you can start, connect to, and stop. Do the steps in order.

### 1C.1 Safety first: budget alert (5 min)

1. Sign in to the AWS console.
2. Search for **Budgets** → **Create budget** → **Use a template** → **Monthly cost budget**.
3. Amount: for example $50 (raise it later). Enter your email. Create.

You'll get an email at 85% and 100% of the amount. It doesn't stop spending; it warns you.

### 1C.2 Choose one region and stick to it

Top-right region menu → **US West (Oregon) us-west-2**. Instances, disks and key pairs are per region; if something seems to have vanished, check the region first.

### 1C.3 Request GPU quota (5 min, then wait)

New accounts are often allowed **zero** GPU machines.

1. Search for **Service Quotas** → **AWS services** → **Amazon Elastic Compute Cloud (Amazon EC2)**.
2. Search for **Running On-Demand G and VT instances**. The value is in **vCPUs**, not machines.
3. If it's below 8: **Request increase at account level** → enter **8** → submit. The reason can be "Stanford CS230 course project: training a pose-estimation model on a single g5.xlarge."

Approval can take from minutes to a couple of days. While waiting, do 1A and 1B.

### 1C.4 Create a key pair (5 min)

1. **EC2** → left menu **Key Pairs** → **Create key pair**.
2. Name `skier-aws`, type **ED25519**, format **.pem** → **Create**. The file downloads once; you can't download it again.
3. Move it to `C:\Users\<you>\.ssh\skier-aws.pem` (or `~/.ssh/` on Mac/Linux).
4. On Windows, restrict its permissions in PowerShell, or SSH will refuse to use it:

   ```powershell
   icacls "$HOME\.ssh\skier-aws.pem" /inheritance:r /grant:r "$($env:USERNAME):R"
   ```

   On Mac/Linux: `chmod 400 ~/.ssh/skier-aws.pem`.

### 1C.5 Launch the instance (10 min)

**EC2** → **Instances** → **Launch instances**:

| Field | Value |
| --- | --- |
| Name | `skier-gpu` |
| AMI | Search "**Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)**" and pick the latest. It has the NVIDIA driver but no preinstalled PyTorch, which is what we want, since we install pinned versions. |
| Instance type | **g5.xlarge** (A10G 24 GB, about $1.0/h). Cheaper and slower: g4dn.xlarge (T4 16 GB, about $0.53/h). |
| Key pair | `skier-aws` |
| Network settings | Allow SSH traffic from **My IP** (not "Anywhere") |
| Storage | **100 GiB**, type **gp3** (it can be grown later without reinstalling) |

Click **Launch instance**. If you get an error about vCPU limits, the quota from 1C.3 isn't approved yet.

### 1C.6 Connect (10 min)

1. **Instances** → select `skier-gpu` → copy **Public IPv4 DNS** (for example `ec2-…us-west-2.compute.amazonaws.com`).
2. Add to `~/.ssh/config` on your machine:

   ```text
   Host skier-aws
     HostName ec2-REPLACE.us-west-2.compute.amazonaws.com
     User ubuntu
     IdentityFile ~/.ssh/skier-aws.pem
   ```

3. Test with `ssh skier-aws`. For editing on the instance, install VS Code's **Remote - SSH** extension → **Connect to Host** → `skier-aws`.

Two things will break the connection later, and both are normal:
- **After every stop/start the public address changes.** Copy the new DNS into `HostName`.
- **If you change networks** (home vs. campus), "My IP" changes. Fix it under **Instances** → **Security** tab → security group → **Edit inbound rules** → SSH source **My IP** → Save.

### 1C.7 Install the environment on the instance (30–45 min)

```bash
nvidia-smi                                   # must show the GPU
sudo apt-get update
sudo apt-get install -y python3.10-venv tmux ffmpeg git unzip
python3.10 -m venv ~/venvs/skier
source ~/venvs/skier/bin/activate
pip install --upgrade pip wheel

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2"
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install "mmengine>=0.10,<0.11"
pip install mmpose==1.3.2
pip install openmim pandas pyyaml tqdm pytest got10k gdown
pip install "numpy<2"                        # re-pin in case anything upgraded it

pip check
python -c "import numpy, torch, mmcv, mmcv.ops, mmpose; print(numpy.__version__, torch.__version__, torch.cuda.is_available())"
```

- If `mmpose` fails while building **chumpy** or **xtcocotools**, run `pip install --no-build-isolation chumpy xtcocotools` and then retry `pip install mmpose==1.3.2`.
- Don't install `opencv-python-headless` on top of the OpenCV that mmcv already pulled in; two OpenCV variants conflict.
- The torch `cu121` build bundles its own CUDA runtime. The AMI's driver only needs to be new enough, which it is.

Then:

```bash
echo 'source ~/venvs/skier/bin/activate' >> ~/.bashrc
echo 'export SKIER_DATA=$HOME/skier_data'  >> ~/.bashrc
echo 'export SKIER_RESULTS=$HOME/skier_results' >> ~/.bashrc
source ~/.bashrc
mkdir -p $SKIER_DATA $SKIER_RESULTS
```

**GitHub on the instance:** run `ssh-keygen -t ed25519 -C "ruichaoz@stanford.edu"`, paste `~/.ssh/id_ed25519.pub` into GitHub → Settings → SSH keys, then:

```bash
git clone git@github.com:RichardZRC/skier_project.git ~/skier_project
cd ~/skier_project && pip install -e . && python scripts/check_env.py
pip freeze > artifacts/step1/pip-freeze.txt
```

### 1C.8 Daily habits

- **Start:** EC2 → Instances → select → **Instance state → Start**, then update `HostName`.
- **Long jobs:** run them inside `tmux new -s train`; detach with `Ctrl-b d`; reattach with `tmux attach -t train`. Closing your laptop then doesn't kill training.
- **Stop when done:** **Instance state → Stop**. Stopped instances cost only disk (about $8/month for 100 GB).
- **Never "Terminate"** unless you mean it: it deletes the disk, including data and checkpoints.

| Item | Approximate cost |
| --- | --- |
| g5.xlarge running | about $1.0 per hour |
| g4dn.xlarge running | about $0.53 per hour |
| 100 GB gp3 disk | about $8 per month, running or stopped |
| Public IPv4 address while running | about $0.005 per hour |

Prices are approximate and region-dependent; check the AWS pricing page.

**While the quota is pending:** you can install the same pinned stack on the local Windows PC (GTX 1070 Ti, 8 GB). Prebuilt Windows wheels exist for Python 3.10 + torch 2.1.2 cu121 + mmcv 2.1.0. That's enough for the smoke tests and a tiny overfit at batch 8.

---

## 1D. Download data and record provenance `[NET][DATA]`

### Ski 2DPose (no form needed)

On the instance:

```bash
mkdir -p $SKIER_DATA/ski2dpose/archives && cd $SKIER_DATA/ski2dpose/archives
wget https://datasets-cvlab.epfl.ch/2019-ski-2d-pose/ski2dpose_labels.json.zip
wget https://datasets-cvlab.epfl.ch/2019-ski-2d-pose/ski2dpose_images_jpg.zip
wget https://datasets-cvlab.epfl.ch/2019-ski-2d-pose/ski2dpose_videos.zip
wget https://datasets-cvlab.epfl.ch/2019-ski-2d-pose/ski2dpose_info.json
wget https://datasets-cvlab.epfl.ch/2019-ski-2d-pose/ski2dpose_example_loader.py
sha256sum * > SHA256SUMS.txt
cd .. && for z in archives/*.zip; do unzip -l "$z" | head -20; done   # look before extracting
```

Then extract each archive into its own folder (`labels/`, `images/`, `videos/`) and keep `archives/` untouched.

The example loader is downloaded code: **read it, don't run it blindly**. It's useful for learning the image path pattern needed in 1E.

### Pretrained weights

```bash
mkdir -p $SKIER_DATA/pretrained
mim download mmpose --config td-hm_hrnet-w32_8xb64-210e_coco-256x192 --dest $SKIER_DATA/pretrained
sha256sum $SKIER_DATA/pretrained/* >> $SKIER_DATA/pretrained/SHA256SUMS.txt
```

This downloads the config and its matching COCO checkpoint. Record both filenames.

### SiamFC and the SkiTB toolkit

```bash
mkdir -p $SKIER_DATA/vendor && cd $SKIER_DATA/vendor
git clone https://github.com/matteo-dunnhofer/SkiTB-toolkit.git
git -C SkiTB-toolkit rev-parse HEAD
```

The SkiTB README links the SiamFC fork to clone and the Google Drive file for `siamfc_alexnet_e50.pth`. Clone that fork next to the toolkit, record its commit, and download the weights with `gdown` (or download locally and copy them up with `scp`).

### SkiTB data (when access arrives)

The download method depends on the link you receive. Download straight to the instance if possible. Record the size, date and any sharing restrictions. Only `AL/` is needed.

### Download ledger — `docs/data_audit.md` `[MANUAL]`

One row per asset: asset, source URL, date, local path, size, SHA-256, license/terms note, status.
- Ski 2DPose terms: no explicit dataset license on the page; source videos are described as YouTube Creative Commons; cite Bachmann et al. 2019. Don't redistribute images or labels in the repo.
- SkiTB: CC BY-NC 4.0.

---

## 1E. Label audit and inventory `[DATA]`

1. Read the example loader and `unzip -l` listings, and write down the real image path pattern. Put it into `--path-template` / `image_path_for`. This is the one deliberate unknown from 1B.
2. Find the info JSON keys for discipline and view angle and fill them into the inventory.
3. Run `audit_pose_labels.py` with images. Expected totals: 16 videos, 158 segments, 1,982 images. Any difference goes in `docs/data_audit.md` with an explanation.
4. Check `problems.csv`. Investigate each problem type on a few examples; don't silently drop records.

**Done when:** every label resolves to an image (or the exceptions are documented), and the counts are recorded.

---

## 1F. Ground-truth overlays and your review `[DATA][MANUAL]`

1. Choose the pilot videos (see 1B.3) and save `pilot_videos.txt`.
2. Run `draw_gt_overlays.py` for 30 images.
3. Review them with the checklist below (30–45 min).

### Review checklist

You're checking that **our code reads the labels correctly**, not re-auditing a published dataset.

| # | Check | If wrong, the likely cause is |
| --- | --- | --- |
| 1 | Points sit on the skier, not bunched in a corner or consistently shifted | Normalized → pixel conversion, or width/height swapped |
| 2 | Orange (right) points are on the skier's **own** right. A skier facing the camera has their right on the image's left. Check both front and back views. | Joint order or side mapping |
| 3 | Each ski reads tip → toes → heel → tail, with toes and heel near the boot | Ski index order |
| 4 | Pole basket points are at the bottom of the poles | Pole indices |
| 5 | Hollow points are plausible guesses; filled points are clearly on the right spot | Visibility flag read backwards |
| 6 | The dashed box contains the whole body and both ski tips | Reference-box logic |

Write `sample_id + check number + what you saw` in `docs/data_audit.md`. **If the same problem shows up in more than about 2 of the 30 images, it's almost certainly a code bug:** fix the code, not the labels. Never "correct" public labels because they look odd; if a real annotation defect appears, write an objective exclusion rule before any training.

---

## 1G. Video-frame alignment `[DATA][MANUAL]`

1. Find out whether `videos/` holds one clip per segment or whole source videos (file count, durations from `ffprobe`). Fill in `locate_segment`.
2. Run `check_alignment.py` on 1–2 pilot segments.
3. Check that every labeled image has a confident match (`margin` clearly above 0) at a **consistent** offset: near the start, middle and end of the segment. One match isn't enough.
4. `[MANUAL]` Watch `gt_clip.mp4`: the labels should land on the skier at labeled frames, not ahead or behind.

Record the indexing convention (0- or 1-based, any offset) in `docs/data_audit.md`.

**If alignment fails** within this time box: record the exact mismatch and move on. Pose work proceeds unchanged; only the Step 7 temporal comparison becomes a stated limitation.

---

## 1H. Smoke tests `[DATA][GPU]`

1. `check_env.py` on the instance: CUDA available, `mmcv.ops` imports, numpy < 2.
2. `smoke_pose.py` on one pilot image with its tight reference box (from the inventory):
   - 17 finite points near the skier;
   - difficult joints may be wrong;
   - no ski points, because the generic model doesn't predict them.

   Record ms/image and peak GPU memory.
3. `smoke_tracker.py` on one Ski 2DPose pilot video, initialized once at the first labeled frame with the reference box. Look at the first, middle and last frames. This is qualitative only, not a benchmark. Once SkiTB data arrives, repeat on one **training** AL single-camera clip. While doing that, check the toolkit's `--split_file` vs. `--split-file` spelling and the `ST` vs. `SC` folder name against the real data, and note any fix in `docs/decisions.md`.

**Done when:** at least one real pose inference runs on the AWS GPU; the tracker runs on a clip or has a named, specific blocker.

---

## 1I. Readiness decision `[MANUAL]`

Create `docs/step1_readiness.md`:

```markdown
# Step 1 readiness

Date:
Active hours spent:

## Data
- Ski 2DPose images/labels: counts (videos / segments / images):
- Unresolved files or problems:
- Pilot videos (forced into training):
- SkiTB status (requested 2026-10-17):
- License/provenance notes:

## Label understanding
- Schema, left/right, ski order verified on overlays:
- Visibility policy verified:
- Review findings:

## Environment (AWS)
- Instance type / region / AMI:
- GPU / VRAM:
- Python / torch / CUDA build / mmcv / mmengine / mmpose / numpy:
- Checkpoint filenames + SHA-256:
- Pose inference: ms/image, peak memory:
- Tracker smoke test:

## Video alignment
- Segments checked / matched:
- Index/offset convention:
- Remaining blocker:

## Decision
- Ready / ready with limitation / blocked:
- Reason:
- Next action and deadline:
```

| State | Meaning | Next |
| --- | --- | --- |
| **Ready** | Labels draw correctly, pose inference works, alignment verified | Step 2 |
| **Ready with limitation** | Pose is feasible; SkiTB or alignment still pending | Step 2, with a deadline on the pending item |
| **Blocked** | Labels can't be read correctly, or no working environment | Fix that blocker before anything else |

If SkiTB access is still missing at the end of Week 1, apply the main plan's fallback and say so in the report.

---

## End-of-step checklist

- [ ] `[MANUAL]` Budget alert set; GPU quota approved; instance launched and reachable.
- [ ] `[CODE]` Scaffold and all 1B scripts written; synthetic tests pass.
- [ ] `[NET]` Pinned environment installed on AWS; `env.json` and `pip-freeze.txt` saved.
- [ ] `[NET][DATA]` Ski 2DPose, COCO checkpoint, SiamFC code and weights downloaded with checksums; ledger filled.
- [ ] `[DATA]` Image path pattern resolved; audit counts recorded; problems explained.
- [ ] `[MANUAL]` Pilot videos chosen and saved.
- [ ] `[MANUAL]` 30 overlays reviewed against the checklist.
- [ ] `[DATA][MANUAL]` Alignment verified or explicitly marked unresolved.
- [ ] `[DATA][GPU]` Pose smoke test passes on AWS; tracker smoke test run or blocker named.
- [ ] `[MANUAL]` SkiTB status recorded; follow-up done if needed.
- [ ] `[MANUAL]` Readiness decision written.

## References (checked 2026-10-17)

- [EPFL Ski 2DPose](https://www.epfl.ch/labs/cvlab/data/ski-2dpose-dataset/)
- [SkiTB toolkit](https://github.com/matteo-dunnhofer/SkiTB-toolkit)
- [MMPose installation](https://mmpose.readthedocs.io/en/latest/installation.html), [version FAQ](https://mmpose.readthedocs.io/en/latest/faq.html)
- [MMCV 2.1.0 wheels (torch 2.1, cu121)](https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html)
- [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)
- [MMPose v1.3.2 inference API](https://github.com/open-mmlab/mmpose/blob/v1.3.2/mmpose/apis/inference.py)
