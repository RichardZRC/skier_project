# Step 1 guide: setup and feasibility (v2)

**Owner:** Ruichao Zhang — ruichaoz@stanford.edu
**Guide version:** v2, October 17, 2026 (replaces v1; v1 is in git history)
**Time budget:** about 12 active hours (Week 1, starting 2026-10-19)
**Full plan:** [Six-week implementation plan](skier_implementation_plan.md)
**Code spec:** [Step 1 code spec](skier_step_1_code_spec.md) (how to write every 1A/1B module and test)

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

## 1A. Repo scaffold `[CODE]` (+ one-time `[NET]`)

Work in the existing repo (`git@github.com:RichardZRC/skier_project.git`); don't create a new folder. Clone it outside any cloud-sync folder.

Build the scaffold exactly as in [code spec](skier_step_1_code_spec.md) §2: folder tree, `pyproject.toml`, `requirements-pose.txt`, `.gitignore`, `paths.py`, `io_utils.py`, and the `docs/scope.md` / `docs/decisions.md` seeds.

**One-time `[NET]`:** create the light dev environment (spec §2.3), which has `numpy<2`, `opencv-python`, `pillow`, `pyyaml` and `pytest`. After that, all `[CODE]` work and its tests run offline. Tests that need torch or mmpose are marked `stack` and skip here; Step 1 has none.

Two repo rules matter because planning and coding happen on different machines:
- The repo is the only channel between them. Small text evidence (JSON/CSV/TXT) is committed under `results/step1/` (see 1I).
- Dataset images and labels are never committed. Overlays, clips and predictions stay under `$SKIER_RESULTS`.

**Done when:** `pip install -e ".[dev]"` and `pytest` run (zero tests is fine at this point).

---

## 1B. Offline coding block `[CODE]`

Write all Step 1 code without data or internet, following the [code spec](skier_step_1_code_spec.md). The spec gives the module layout, signatures, behaviour and the exact synthetic tests. Facts that only the real files can answer sit in marked `FILL IN` blocks (spec §13), which get filled in during 1E–1H.

| Task | What | Entry point | Spec |
| --- | --- | --- | --- |
| 1B.1 | 24-landmark schema: names, groups, flip pairs, edges, COCO mapping | `configs/ski24_schema.json`, `skier.schema` | §3 |
| shared | Raw label reader, image paths, reference box | `skier.data.ski2dpose_raw`, `skier.geometry` | §4–5 |
| 1B.2 | Label audit and image inventory | `scripts/audit_pose_labels.py` | §6 |
| 1B.3 | Ground-truth overlays and contact sheet | `scripts/draw_gt_overlays.py` | §7 |
| 1B.4 | Video-frame alignment and GT clip | `scripts/check_alignment.py` | §8 |
| 1B.5 | Environment check (`--strict` on AWS) | `scripts/check_env.py` | §9 |
| 1B.6 | Pretrained pose smoke test (MMPose's own pipeline) | `scripts/smoke_pose.py` | §10 |
| 1B.7 | SiamFC tracker smoke test | `scripts/smoke_tracker.py` | §11 |

Every tool writes to `--out`, which defaults to `$SKIER_RESULTS/step1/<tool>/` (spec §1).

**Done when:** spec §14 passes: `pytest` is green in the light environment, and every script's `--help` works without torch installed.

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
cd ~/skier_project && pip install -e ".[dev]" && pytest
mkdir -p $SKIER_RESULTS/step1/env
python scripts/check_env.py --strict          # writes $SKIER_RESULTS/step1/env/env.json; exit code 1 = something to fix
pip freeze > $SKIER_RESULTS/step1/env/pip-freeze.txt
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

1. Read the example loader and the `unzip -l` listings, and write down the real image path pattern. Put it into `IMAGE_PATH_TEMPLATE` (code spec §13). This is the main deliberate unknown from 1B.
2. Run the audit once with `--info` and no images, then read `info_preview` in `label_summary.json` to find the discipline and view-angle keys. Fill in `info_for` (spec §13).
3. Run the full audit with images. Expected totals: 16 videos, 158 segments, 1,982 images (`matches_expected` in the summary). Any difference goes in `docs/data_audit.md` with an explanation.

   ```bash
   cd ~/skier_project
   python scripts/audit_pose_labels.py $SKIER_DATA/ski2dpose/labels/<labels>.json \
     --info $SKIER_DATA/ski2dpose/archives/ski2dpose_info.json \
     --images-root $SKIER_DATA/ski2dpose/images
   # → $SKIER_RESULTS/step1/audit/{label_summary.json,problems.csv} and data/manifests/pose_inventory.csv
   ```

4. Check `problems.csv`. Investigate each problem type on a few examples; don't silently drop records.

**Done when:** every label resolves to an image (or the exceptions are documented), and the counts are recorded.

---

## 1F. Ground-truth overlays and your review `[DATA][MANUAL]`

1. Choose 2–3 **pilot videos**: large, clearly visible skiers, different race types, and none of the follow-cam videos. Save their IDs in `data/manifests/pilot_videos.txt`, one per line. They are forced into **training** in Step 2 (`forced_train`), so looking at them closely never contaminates val or test.
2. Draw 30 overlays:

   ```bash
   python scripts/draw_gt_overlays.py --inventory data/manifests/pose_inventory.csv \
     --labels $SKIER_DATA/ski2dpose/labels/<labels>.json --images-root $SKIER_DATA/ski2dpose/images \
     --videos-file data/manifests/pilot_videos.txt --n 30 --seed 230
   # → $SKIER_RESULTS/step1/overlays/ (PNGs stay there; only overlay_ids.txt is synced in 1I)
   ```

3. Review them with the checklist below (30–45 min). Copy the overlays to your own machine with `scp` or open them through VS Code Remote.

**Legend:**
- Orange = skier's right; cyan = left; white = head/neck.
- Filled circle = visible (`visible = 1`); hollow circle = occluded guess (`visible = 0`). The number beside each point is its schema index.
- Body edges are thin, skis thick, poles dotted. A dashed edge means one of its endpoints is a guess.
- Dashed yellow box = reference box.

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

1. Find out whether `videos/` holds one clip per segment or whole source videos (file count, durations from `ffprobe`). Fill in `locate_segment` (code spec §13).
2. Run `check_alignment.py` on 1–2 pilot segments, with `--labels` so it also writes `gt_clip.mp4`. Before `locate_segment` is filled in, pass `--video FILE --start-frame N` by hand. Output goes to `$SKIER_RESULTS/step1/alignment/<video>__<segment>/`.
3. Check that every labeled image has a confident match (`margin` clearly above 0) at a **consistent** offset: near the start, middle and end of the segment. One match isn't enough.
4. `[MANUAL]` Watch `gt_clip.mp4`: the labels should land on the skier at labeled frames, not ahead or behind.

Record the indexing convention (0- or 1-based, any offset) in `docs/data_audit.md`.

**If alignment fails** within this time box: record the exact mismatch and move on. Pose work proceeds unchanged; only the Step 7 temporal comparison becomes a stated limitation.

---

## 1H. Smoke tests `[DATA][GPU]`

1. `check_env.py --strict` on the instance (already run in 1C.7): CUDA available, `mmcv.ops` imports, numpy < 2.
2. `smoke_pose.py` on one pilot image with its tight reference box (from the inventory):
   - 17 finite points near the skier;
   - difficult joints may be wrong;
   - no ski points, because the generic model doesn't predict them.

   ms/image and peak GPU memory are saved in `$SKIER_RESULTS/step1/smoke_pose/smoke_pose.json`.
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

### Sync evidence to the repo `[DATA][NET]`

The planning machine sees only what is in git. Copy the small text outputs (and nothing else) into the repo, then push:

```bash
cd ~/skier_project
mkdir -p results/step1
rsync -a --prune-empty-dirs --include='*/' --include='*.json' --include='*.csv' --include='*.txt' \
      --exclude='*' $SKIER_RESULTS/step1/ results/step1/
git add results/step1 data/manifests docs
git status        # check: no .png / .mp4 / .npz, no dataset images or label files
git commit -m "Step 1 evidence" && git push
```

That brings over `env.json`, `pip-freeze.txt`, `label_summary.json`, `problems.csv`, `overlay_ids.txt`, `alignment.csv`, `alignment_summary.json`, `smoke_pose.json` and `boxes.csv`. None of them contain label coordinates (code spec §1).

---

## End-of-step checklist

- [ ] `[MANUAL]` Budget alert set; GPU quota approved; instance launched and reachable.
- [ ] `[CODE]` Scaffold and all 1B scripts written; code spec §14 acceptance passes.
- [ ] `[NET]` Pinned environment installed on AWS; `check_env.py --strict` passes; `env.json` and `pip-freeze.txt` saved.
- [ ] `[NET][DATA]` Ski 2DPose, COCO checkpoint, SiamFC code and weights downloaded with checksums; ledger filled.
- [ ] `[DATA]` Image path pattern resolved; audit counts recorded; problems explained.
- [ ] `[MANUAL]` Pilot videos chosen and saved.
- [ ] `[MANUAL]` 30 overlays reviewed against the checklist.
- [ ] `[DATA][MANUAL]` Alignment verified or explicitly marked unresolved.
- [ ] `[DATA][GPU]` Pose smoke test passes on AWS; tracker smoke test run or blocker named.
- [ ] `[MANUAL]` SkiTB status recorded; follow-up done if needed.
- [ ] `[MANUAL]` Readiness decision written.
- [ ] `[NET]` Evidence synced to `results/step1/` and pushed (1I).

## References (checked 2026-10-17)

- [EPFL Ski 2DPose](https://www.epfl.ch/labs/cvlab/data/ski-2dpose-dataset/)
- [SkiTB toolkit](https://github.com/matteo-dunnhofer/SkiTB-toolkit)
- [MMPose installation](https://mmpose.readthedocs.io/en/latest/installation.html), [version FAQ](https://mmpose.readthedocs.io/en/latest/faq.html)
- [MMCV 2.1.0 wheels (torch 2.1, cu121)](https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html)
- [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)
- [MMPose v1.3.2 inference API](https://github.com/open-mmlab/mmpose/blob/v1.3.2/mmpose/apis/inference.py)
