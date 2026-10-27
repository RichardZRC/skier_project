# Data audit — SkiTB (alpine, `AL/`)

Evidence for guide steps 1D–1E for SkiTB (the tracking data). Ski 2DPose (the pose data) has its own file: `data_audit_ski2dpose.md`. Findings are facts about the files, not fixes. Never "correct" public labels here.

## 1D. Download ledger

- **Received:** 2026-10-18, through the access form (University of Udine). It came already extracted, as `AL/`, `FS/`, `JP/` and `README.md`. There's no archive, so there's no published checksum to compare against.
- **Location:** moved from `~/Downloads/SkiTB` to `~/skier_data/skitb/`, outside iCloud (D5). Code takes this folder as `--skitb-root`.
- **Integrity:** recorded as a tree digest instead (defined below). After copying to AWS, rerun the audit script there; the same digest means the copy is complete.

| Asset | Files | Bytes | SHA-256 | Status |
| --- | ---: | ---: | --- | --- |
| `AL/` sequence folders AL0001–AL0100 | 219,217 | 15,537,744,392 | tree `f4d143406d2e4d31c4bab08e5a129391f02812d216f5d86cbe3c1ada1f99ed1a` | OK, used |
| `AL/AL_data.csv` | 1 | 27,259 | `0b2e3579a63553c824eb713394df902239e843c10348ad4036d88e75403fc11a` | Metadata, not used by code |
| `AL/AL_train_val_test_athlete_60-40.json` | 1 | 1,030 | `0a15c719bad0dd98e854c267819b6d2895cdb240401daa8d162ae2d4b500926d` | **The split we use (D8)** |
| `AL/AL_train_test_athlete_60-40.json` | 1 | 1,021 | `b9dc5b315e0bf7ace4dfc8935925b6a8e9c1eae29325dfc4ee922bda5ccb1485` | Not used |
| `AL/AL_train_{val_,}test_{course,date}_60-40.json` | 4 | 1,022–1,032 | in `results/step1/skitb_audit/` | Not used |
| `AL/AL_visual_attributes.csv` | 1 | 16,716 | `aea10dc5e45a9bea1816ed07d7f6a0485ccb3ae0a2ae69ba467cd606126b7445` | Not used (see Layout) |
| `README.md` | 1 | 6,923 | `e0e2a001f3d3161c35fd3c1cf27fa40058c828bca7bbc29b3385b350a668f4cc` | Read only |
| `FS/` (4.7 GB), `JP/` (2.0 GB) | | | | Not used. Don't copy them to AWS. |
| SkiTB toolkit (code), `$SKIER_DATA/vendor/SkiTB-toolkit` | | | commit `b402f9a5e59f0c3a3edb2f46a68d1522b6c03a0d` | Cloned on the laptop 2026-10-26 |
| SiamFC fork `matteo-dunnhofer/siamfc-pytorch`, `$SKIER_DATA/vendor/siamfc_pytorch` | | | commit `a87f140ab96a3c78c26d54ece42bf6f304bf06df` | Cloned 2026-10-26; `siamfc.TrackerSiamFC` (adapter filled in) |
| `siamfc_alexnet_e50.pth` (Google Drive id `1UdxuBQ1qtisoWYFZxLgMFJ9mJtGVw6n4`, from the toolkit README) | 1 | 9,359,658 | `2b3df5c66c814f3a3e0ade83c8fd68ff9c714574a4dc17292317ee11f7e717f6` | OK. GOT-10k-pretrained: the toolkit README calls it "the GOT-10k pre-trained SiamFC", and the fork's `tools/train.py` trains on GOT-10k. |

**The tree digest:**
- Per sequence: SHA-256 over the lines `relative/path<TAB>sha256(file)`, sorted by path, with `.DS_Store` skipped.
- Tree: SHA-256 over the lines `sequence<TAB>sequence digest`, sorted, followed by `name<TAB>sha256` for the 8 `AL_*` metadata files, sorted by name.
- Per-sequence digests: `results/step1/skitb_audit/skitb_al_sequences.csv`. Script: `results/step1/skitb_audit/audit_skitb_al.py`, a one-off that isn't part of the package. Run it with `python -I audit_skitb_al.py <skitb root> <out dir>`.

**Terms:**
- The annotations are **CC BY-NC 4.0**: give credit, research use only, no commercial use.
- The frames come from YouTube broadcasts; the links are in `AL_data.csv`.
- Cite Dunnhofer et al., WACV 2024.
- Frames and annotation files are never committed to this repo. The committed evidence holds only counts and digests.

The total frame count across `AL/`, `FS/` and `JP/` is 352,978 JPGs.

**Laptop copy (2026-10-26):** Ruichao copied the whole `skitb/` folder (`AL/`, `FS/`, `JP/`) from the Mac into WSL `~/skier_data/skitb/`. The copy brought **363,486 macOS AppleDouble sidecars** (`._*`, the signature `00 05 16 07` on every one), which broke the audit script. Claude deleted only those files. Then `audit_skitb_al.py` was rerun on the laptop: **all 100 per-sequence digests and the tree digest `f4d14340…` match**; 219,217 files, 15,537,744,392 bytes, 0 problems (`results/step1/skitb_audit/laptop_skitb_al_summary.json`). 4 `.DS_Store` files remain; the digest skips them by design.

## 1E. Layout and audit

**Layout** (matches the dataset README; checked file by file):

| Item | Fact | Used in code (`data/skitb.py`) |
| --- | --- | --- |
| Sequence | `AL/<ID>/` with `frames/`, `MC/` (full multi-camera run) and `SC/` (single-camera clips) | `load_skitb_clips` |
| Clip folders | `SC/<camera index>/`, named `0` to `n−1`. It's **`SC`**, not `ST`. | `SINGLE_CAMERA_DIR`; clips sorted numerically |
| `boxes.txt` | `x,y,w,h` per frame, comma-separated, in pixels | `parse_boxes_txt` (→ `xyxy`) |
| `frames.txt` | **Bare frame indices** into the source video (e.g. `197`), not filenames | `frame_file`: `197` → `00197.jpg` |
| `frames/` | `NNNNN.jpg`, a 5-digit frame index; exactly the frames listed in `MC/frames.txt` | `Clip.frames_dir` |
| `visibilities.txt` | `0`/`1`. `1` means more than 50% of the skier is visible. | `Clip.visible` |
| `MC/` | The same files plus `cameras.txt` | Not used (the core scope is single-camera) |
| Split files | Six JSONs, each `{"train": [...], ("val": [...],) "test": [...]}` of sequence IDs | `sequences_for_partition` |
| `AL_data.csv` | One row per sequence: athlete, nation, gender, course, discipline (SL/GS/SG/DH), date, weather, YouTube link, race time | Audit only |
| `AL_visual_attributes.csv` | Semicolon-separated attributes per single-camera clip, but only for the test set of the **date** split, not ours | Not used |

**Audit result** (`results/step1/skitb_audit/`, run 2026-10-18): **0 format problems.**
- Each of the six split files lists all 100 sequences exactly once.
- `train_test_athlete`'s test set equals `train_val_test_athlete`'s test set, and its train set equals our train plus val.
- Every single-camera clip:
  - has equal line counts in its three files;
  - covers consecutive frames;
  - has its frames in `MC/` with the same box and visibility;
  - matches its folder name with the camera index;
  - has every frame file present.
- Every box has positive width and height, and none extends past the image.
- **Every clip's first frame is visible,** so the tracker always initializes on the clip's first frame.
- `MC/` frame indices have gaps in some sequences (between cameras); single-camera clips never do.
- In AL0016, AL0034, AL0066 and AL0079, `MC/` holds frames from a camera that has no `SC/` folder (e.g. AL0079 has cameras 0–12 and 14 and folders 0–12). Those frames aren't in any clip; nothing to fix.
- **Frame sizes:** one per sequence, but not the same across sequences.
  - 1280×720: 90 sequences
  - 1152×720: 5
  - 1274×720, 1280×716, 1280×718, 640×360, 634×360: 1 each

  The replay must take its output size from each clip.
- The real reader loads the val partition (75 clips, 13,180 frames) with 0 missing frame files.

**Single-camera clips per partition** (athlete split):

| Partition | Sequences | Clips | Frames | Clip length min / median / max | Visible frames | Sequences SL / GS / SG / DH |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| train | 53 | 554 | 110,595 | 1 / 187 / 595 | 98.4% | 14 / 17 / 13 / 9 |
| val | 6 | 75 | 13,180 | 28 / 180 / 394 | 98.6% | 2 / 1 / 1 / 2 |
| test | 41 | 471 | 89,295 | 15 / 170 / 646 | 98.8% | 11 / 8 / 11 / 11 |

**Short clips:**
- train: one clip of 1 frame, 8 under 25 frames;
- test: 2 under 25 frames, the shortest 15;
- val: the shortest is 28.

A 1-frame clip has nothing to score after initialization.

## Findings that affect Steps 3, 4 and 8

1. **The "athlete" split isn't fully athlete-disjoint by the metadata.**
   - Five athletes appear in both train and test, covering **10 of the 41 test sequences**:

     | Athlete | Train | Test |
     | --- | --- | --- |
     | Vlhova | AL0007, AL0062 | AL0005, AL0037, AL0041 |
     | Odermatt | AL0011, AL0014, AL0036 | AL0065, AL0066 |
     | Hirscher | AL0050, AL0058 | AL0017, AL0039, AL0045 |
     | Gut-Behrami | AL0023 | AL0031 |
     | Paris | AL0075, AL0100 | AL0097 |

   - Val shares no athlete with train or test.
   - AL0058 lists Hirscher with gender `W`, presumably a metadata typo.
   - **Effect: no leakage.** We never train the tracker, and `siamfc_alexnet_e50.pth` is GOT-10k-pretrained, not trained on skiing (`PROJECT_NOTES.md`, verified 2026-10-17). Reconfirmed 2026-10-26 from the toolkit README and the fork's `tools/train.py`.
   - We tune only the replay, on val, and val shares no athlete with test.
   - D8's "athlete-disjoint" describes the file's intent, not the metadata; the report should say "the official athlete split".
2. **The frame rate isn't in the files, and it differs between sequences.**
   - (last − first `MC/` frame) ÷ race time gives about 30 for 81 sequences, about 25 for 18, and 38 for AL0098.
   - It's only an estimate: the `MC/` span isn't exactly the race.
   - `run_tracker.py --fps` takes one value for all clips, but the failure episodes (≥ 0.5 s) and all replay timing (τ in seconds, acceleration) depend on it.
   - **Open:** get the real per-sequence frame rate (from the toolkit or the source videos' metadata) and pass it per clip.
   - **2026-10-26: the toolkit doesn't have it either.** At commit `b402f9a`, its only "fps" values are tracker speed (`speed_fps = 1 / time per frame` in `toolkit/experiments/skitb.py`), not video frame rate, and its metrics are frame-based. Remaining sources: the source videos (YouTube links in `AL_data.csv`) or the SkiTB paper. The README's `--split-file` is `--split_file` in `example_skitb.py`, as noted on 2026-10-17.
   - **The paper** ([arXiv 2312.09723](https://arxiv.org/html/2312.09723v1), §4): "The videos have a framerate of 25/30 FPS". Its own time metrics use 30 frames = 1 s ("1 frame (0.03s) … 30 frames (1s)"). It gives no per-video rate.
   - **Per-sequence estimate** (`results/step3/skitb_fps_estimate.csv`, script `skitb_fps_estimate.py`): the MC span ÷ `AthleteTime` falls in two well-separated groups. 18 sequences lie at 23.4–25.8 and 82 at 29.1–31.5, plus AL0098 at 37.9. Nothing lies between 25.83 and 29.08. Snapped to the nearer of 25 and 30: val has 1 × 25 (AL0027) and 5 × 30; test has 8 × 25 and 33 × 30. **Proposed as D22.**
3. **Mixed frame sizes** (see the audit result above).
