# GPU laptop setup: Windows + WSL2, RTX 4070 (D17)

This is the project's GPU machine for now. AWS (guide §1C) stays in the plan for heavier or parallel work once the account is verified. The setup takes about 1–1.5 hours, about half of it waiting for downloads.

**What runs where:**

| Machine | Jobs |
| --- | --- |
| **Mac** | Coding, the light unit tests, docs, the overlay review (1F), the alignment check (1G), the split freeze. Nothing is installed there (D13). |
| **Laptop (now)** | The full test gate, the smoke tests and MMPose cross-check, E0, E1 (overfit, profile, full), E2–E4, evaluation and comparison, and the tracker and replay runs. |
| **AWS (later)** | Heavier or parallel work, such as extra seeds, or anything that doesn't fit 8 GB or runs too slowly on the laptop. |

**Why WSL2:** it runs real Ubuntu inside Windows, with full access to the GPU. Every command below is then the same Linux command as on AWS, and the pinned Linux wheels install unchanged.

**How we work:** you run the commands on the laptop and paste the output to Claude. Code moves through GitHub: Claude pushes from the Mac, and you `git pull` on the laptop. Small text evidence can be pushed from the laptop (Part 7).

---

## Part 1: Windows (once)

### 1.1 NVIDIA driver
1. Update the driver with the **NVIDIA app** (or from nvidia.com → Drivers). The Game Ready and Studio drivers both work, and they already include WSL support.
2. In **PowerShell**, run `nvidia-smi`. It should list the **RTX 4070 Laptop GPU**, its memory (probably 8 GB), and "CUDA Version" **12.1 or higher**.

### 1.2 WSL2 with Ubuntu 22.04
1. Open **PowerShell as Administrator**: Start menu → type "PowerShell" → right-click → *Run as administrator*.
2. Run:
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```
3. Restart when asked. An **Ubuntu** window then opens and asks for a new Linux **username and password**. They're separate from your Windows login, and you'll need the password for `sudo`.
4. Back in PowerShell, check:
   ```powershell
   wsl --update
   wsl -l -v          # Ubuntu-22.04 must show VERSION 2
   ```
   If it shows VERSION 1, run `wsl --set-version Ubuntu-22.04 2`.
5. **From now on, open Ubuntu** from the Start menu, or from a Windows Terminal tab. Every remaining command runs there.

### 1.3 Disk space
WSL keeps its files on drive C:. Make sure C: has **at least 40 GB free**: about 6 GB for the software, 1 GB for Ski 2DPose, about 7 GB later for the SkiTB subset, plus results and checkpoints.

---

## Part 2: Ubuntu inside WSL

### 2.1 Check the GPU
```bash
nvidia-smi        # the same GPU as in Windows
```
**Never install an NVIDIA driver or the CUDA toolkit inside WSL.** The Windows driver provides the GPU, and PyTorch's `cu121` wheels bundle their own CUDA runtime.

### 2.2 System packages
```bash
sudo apt-get update
sudo apt-get install -y python3.10-venv python3.10-dev build-essential git unzip tmux ffmpeg curl
```

### 2.3 Python environment with the pinned stack (about 15 min)
Run this inside `tmux new -s setup`, so a closed window doesn't interrupt it. These are the same versions as guide §1C.7 and `requirements-pose.txt`.
```bash
python3.10 -m venv ~/venvs/skier
source ~/venvs/skier/bin/activate
pip install --upgrade pip wheel
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2"
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install "mmengine>=0.10,<0.11"
pip install mmpose==1.3.2
pip install mmdet==3.2.0                     # mmpose can't load models without it (D18)
pip install openmim pandas pyyaml tqdm pytest got10k gdown
pip install "numpy<2" "opencv-python<4.12"   # re-pin; OpenCV 5.x needs numpy>=2 (D18)
pip check
python -c "import numpy, torch, mmcv, mmcv.ops, mmpose; print(numpy.__version__, torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expect something like `1.26.4 2.1.2+cu121 True NVIDIA GeForce RTX 4070 Laptop GPU`.

- If `mmpose` fails while building **chumpy** or **xtcocotools**: run `pip install --no-build-isolation chumpy xtcocotools`, then retry `pip install mmpose==1.3.2`.
- Don't install `opencv-python-headless` on top; two OpenCV variants conflict.
- `pip check` must say "No broken requirements found". If it names `opencv-python` and numpy, the OpenCV re-pin above didn't run.
- Without `mmdet`, every `mmpose.apis` import fails with `No module named 'mmdet'`, and 2 HRNet tests fail in the gate (D18).

### 2.4 Settings
```bash
echo 'source ~/venvs/skier/bin/activate' >> ~/.bashrc
echo 'export SKIER_DATA=$HOME/skier_data' >> ~/.bashrc
echo 'export SKIER_RESULTS=$HOME/skier_results' >> ~/.bashrc
source ~/.bashrc
mkdir -p $SKIER_DATA $SKIER_RESULTS
```
**Keep the repo, data and results in the Linux home folder (`~`), never under `/mnt/c/...`.** Windows-drive access from WSL is many times slower, which would starve training of data.

### 2.5 GitHub access and the code
```bash
ssh-keygen -t ed25519 -C "ruichaoz@stanford.edu"     # press Enter at every prompt
cat ~/.ssh/id_ed25519.pub
```
Copy the printed line into GitHub → **Settings → SSH and GPG keys → New SSH key**. Then:
```bash
ssh -T git@github.com                                 # "Hi RichardZRC! ..." means it works
git clone git@github.com:RichardZRC/skier_project.git ~/skier_project
cd ~/skier_project && pip install -e ".[dev]"
git config user.name "Ruichao Zhang" && git config user.email "ruichaoz@stanford.edu"
```

### 2.6 The gate: run this before anything else
```bash
cd ~/skier_project
pytest -q                                             # expect: 257 passed
mkdir -p $SKIER_RESULTS/step1/env
python scripts/check_env.py --strict                  # expect: exit code 0 (every check OK, including the GPU)
pip freeze > $SKIER_RESULTS/step1/env/pip-freeze.txt
```
If anything fails, stop and paste the output to Claude. In particular, a failing `mmcv_ops_import` or `gpu_conv` check matters.

---

## Part 3: Data and weights (in the Linux home folder)

### 3.1 Ski 2DPose (about 860 MB with the videos)
```bash
mkdir -p $SKIER_DATA/ski2dpose/archives && cd $SKIER_DATA/ski2dpose/archives
B=https://datasets-cvlab.epfl.ch/2019-ski-2d-pose
for f in ski2dpose_labels.json.zip ski2dpose_images_jpg.zip ski2dpose_videos.zip ski2dpose_info.json; do curl -fLO "$B/$f"; done
sha256sum *                 # must match docs/data_audit_ski2dpose.md exactly
cd .. && mkdir -p labels images videos
unzip -q archives/ski2dpose_labels.json.zip -d labels
unzip -q archives/ski2dpose_images_jpg.zip -d images
unzip -q archives/ski2dpose_videos.zip -d videos     # the videos are for the tracker and replay work
```
Expected totals: 16 videos, 158 segments, 1,982 images.

### 3.2 COCO HRNet-W32 config and weights
```bash
mim download mmpose --config td-hm_hrnet-w32_8xb64-210e_coco-256x192 --dest $SKIER_DATA/pretrained
ls $SKIER_DATA/pretrained   # one .py config, one .pth checkpoint
```

### 3.3 Later, for the tracker work
- **SiamFC fork and `siamfc_alexnet_e50.pth`:** clone the SkiTB toolkit (`git clone https://github.com/matteo-dunnhofer/SkiTB-toolkit.git`). Its README links the SiamFC fork and the weights (Google Drive, so use `gdown`). Paste the fork's tracker class to Claude, so the `FILL IN` block in `tracking/siamfc_adapter.py` can be completed.
- **SkiTB subset:**
  - Copy from the Mac's `~/skier_data/skitb/AL/` only the 6 val and 41 test sequences, plus one train sequence (for example `AL0001`), and the `AL_*` files. A USB drive or a cloud drive both work.
  - Put them in `$SKIER_DATA/skitb/AL/`.
  - Verify them with `results/step1/skitb_audit/audit_skitb_al.py` (compare the per-sequence digests with `skitb_al_sequences.csv`).

---

## Part 4: First GPU session (the trial)

```bash
cd ~/skier_project && git pull
P=$SKIER_DATA/pretrained
CFG=$P/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py
COCO=$(ls $P/td-hm_hrnet-w32_8xb64-210e_coco-256x192-*.pth)

# one pilot image with its labeled skier box
python scripts/smoke_pose.py --config $CFG --checkpoint $COCO \
  --image $SKIER_DATA/ski2dpose/images/Images_jpg/5UHRvqx1iuQ/0/5UHRvqx1iuQ_0_00010.jpg \
  --box 1225 194 1726 919 --device cuda:0

# our pipeline vs. MMPose on 10 pilot images: must agree within 2 px (D3, D12)
python scripts/crosscheck_mmpose.py --config $CFG --checkpoint $COCO --device cuda:0
```

**Send Claude:**
- the import line from 2.3;
- the last lines of `pytest -q`;
- the `check_env.py --strict` output;
- the `sha256sum` output;
- the smoke-test output;
- the cross-check output.

---

## Part 5: Training and evaluation (after the split is frozen on the Mac)

The split and the crop-error set are frozen and committed **on the Mac**; the laptop only pulls them. Run long jobs inside `tmux`: start with `tmux new -s train`, detach with `Ctrl-b d`, and reattach with `tmux attach -t train`.

```bash
cd ~/skier_project && git pull
SPLIT=data/manifests/pose_split_v1.json
# (CFG and COCO as in Part 4)

# E0: the unchanged COCO model on val
python scripts/eval_generic_pose.py --config $CFG --checkpoint $COCO --split $SPLIT --device cuda:0

# E1, step by step: each must succeed before the next
python scripts/train_pose.py --split $SPLIT --hrnet-config $CFG --checkpoint $COCO --device cuda:0 --overfit 32
python scripts/train_pose.py --split $SPLIT --hrnet-config $CFG --checkpoint $COCO --device cuda:0 --profile-steps 200
```
**Stop after the profile and send Claude `$SKIER_RESULTS/pose/E1/profile.json`.** It shows the seconds per step, the share of time spent waiting for data, the peak GPU memory and the estimated length of the full run. If 8 GB is too tight for batch 16, the batch size changes, and that is a recorded decision applying to E1–E4 alike, before any full run.

```bash
# full E1 (inside tmux)
python scripts/train_pose.py --split $SPLIT --hrnet-config $CFG --checkpoint $COCO --device cuda:0

# E2–E4: same command, other config
for n in 2 3 4; do
  python scripts/train_pose.py --split $SPLIT --hrnet-config $CFG --checkpoint $COCO --device cuda:0 \
    --config configs/pose/e$n.yaml --experiment E$n
done

# evaluate on val with the crop-error set, then compare (Step 6, D16)
for n in 1 2 3 4; do
  python scripts/eval_pose.py --checkpoint $SKIER_RESULTS/pose/E$n/best.pt --hrnet-config $CFG --split $SPLIT \
    --crop-error-set data/manifests/crop_error_set_v1.json --device cuda:0
done
python scripts/compare_experiments.py E0=$SKIER_RESULTS/pose/E0/val/metrics.json \
  E1=$SKIER_RESULTS/pose/E1/eval/val/metrics.json E2=$SKIER_RESULTS/pose/E2/eval/val/metrics.json \
  E3=$SKIER_RESULTS/pose/E3/eval/val/metrics.json E4=$SKIER_RESULTS/pose/E4/eval/val/metrics.json
```
**Never pass `--partition test`** (or `--final-test`) before Step 8.

**Same machine for everything compared (D17):** E1–E4, and any seeds in the same table, are trained on the same kind of machine with the same batch size. If extra seeds move to AWS later, E1 gets retrained there too.

---

## Part 6: Habits for long runs on a laptop

- **Plugged in**, with the Windows power mode set to **Best performance**.
- **Sleep: Never** while plugged in. Lid close: **Do nothing**, or keep the lid open.
- **Pause Windows Update** during runs, so it can't restart the laptop.
- **Keep the Ubuntu window open.** Closing every WSL window can shut WSL down, and with it the run, even inside `tmux`.
- Keep the vents clear. Watch the temperature and clock with `nvidia-smi -l 5` in a second tmux window; laptops slow down when they get hot.
- **To see images and results from Windows:** open `\\wsl$\Ubuntu-22.04\home\<linux-user>\skier_results` in File Explorer (for example the training's `val_overlays`).

---

## Part 7: Getting results back to the repo

- **Simplest:** paste the outputs to Claude.
- **Or push small text evidence from the laptop**, never images, labels, checkpoints or `.npz` files:
  ```bash
  cd ~/skier_project && git pull
  mkdir -p results/step1/laptop
  cp $SKIER_RESULTS/step1/env/env.json $SKIER_RESULTS/step1/env/pip-freeze.txt results/step1/laptop/
  git add results/step1/laptop && git commit -m "Laptop environment evidence" && git push
  ```

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `nvidia-smi` not found inside Ubuntu | Update the Windows NVIDIA driver, run `wsl --update` in PowerShell, then reopen Ubuntu |
| `torch.cuda.is_available()` is `False` | Same as above. Also check that `torch.__version__` ends in `+cu121`, not `+cpu`. |
| `check_env.py` reports `mmcv_ops_import` or `gpu_conv` failing | Paste the full output to Claude |
| `CUDA out of memory` during training | Stop and send the profile; the batch size is a recorded decision (Part 5) |
| Training spends most of its time waiting for data | Make sure the data is in `~/skier_data`, not under `/mnt/c`. If it still waits, ask Claude before using `--cache-dir`: the first run that uses a cache builds it for that run's augmentation, so it has to be built with `e4.yaml` to cover E1–E4. |
| `Permission denied (publickey)` from GitHub | Add `~/.ssh/id_ed25519.pub` to GitHub (2.5) |
| A run stopped overnight | Sleep, a Windows Update restart, or WSL shutting down (Part 6). Resume with `--resume $SKIER_RESULTS/pose/E1/last.pt`. |
