# Data audit — Ski 2DPose

Evidence for guide steps 1D–1G for Ski 2DPose (the pose data). SkiTB (the tracking data) has its own file: `data_audit_skitb.md`. One section per step; findings are facts about the files, not fixes. Never "correct" public labels here.

## 1D. Download ledger

Downloaded 2026-10-17 to `~/skier_data/ski2dpose/archives/` (Ruichao's Mac, outside iCloud). Checksums are in `archives/SHA256SUMS.txt`; `shasum -a 256 -c SHA256SUMS.txt` passed after extraction.

| Asset | Source URL | Bytes | SHA-256 | Status |
| --- | --- | ---: | --- | --- |
| `ski2dpose_labels.json.zip` | https://datasets-cvlab.epfl.ch/2019-ski-2d-pose/ski2dpose_labels.json.zip | 833,399 | `8110b682c41603eccff075c85bda32e3a10e05655a7acc7ead68ae9829106ae4` | OK, extracted to `labels/` |
| `ski2dpose_images_jpg.zip` | https://datasets-cvlab.epfl.ch/2019-ski-2d-pose/ski2dpose_images_jpg.zip | 341,041,634 | `79e0df792249f7698755d1fe5127fc912a48d2af158f8462ebd1ce409e0467d0` | OK, extracted to `images/` |
| `ski2dpose_videos.zip` | https://datasets-cvlab.epfl.ch/2019-ski-2d-pose/ski2dpose_videos.zip | 519,729,922 | `dd362323a33c2e4bdd96b9134b3a68470a3cb54dd0070b2679a487b478041ab4` | OK, extracted to `videos/` |
| `ski2dpose_info.json` | https://datasets-cvlab.epfl.ch/2019-ski-2d-pose/ski2dpose_info.json | 40,499 | `8e7e20589f6961962ccac6a46d7550850256f5eb4101d4698259663327e342f7` | OK, used in place |
| `ski2dpose_example_loader.py` | https://datasets-cvlab.epfl.ch/2019-ski-2d-pose/ski2dpose_example_loader.py | 8,817 | `4334e927a695e42ad150f3c505ca8319375c2bf0ff2f233285c0da4e65bb5ab0` | Read only, never run |
| COCO HRNet-W32 config + checkpoint | via `mim download` | | | Pending: needs the GPU stack (AWS) |

**Terms:** no explicit dataset license on the page. The source videos are described as YouTube Creative Commons. Cite Bachmann et al., 3DV 2019. Images and labels are never committed to this repo.

**Sizes vs. the EPFL page:**
- The page's "5.1 MB" for labels is the **unzipped** JSON (5,107,158 bytes).
- The images zip is 341 MB, against the 352 MB listed. Its integrity test passes, and it holds exactly 1,982 JPGs, one per label. So it's complete; the page figure appears to be rounded differently.

## 1E. Layout and label audit

**Layout:**

| Item | Fact | Used in code |
| --- | --- | --- |
| Labels | `labels/ski2dpose_labels.json`; image IDs are full stems (`<video>_<segment>_NNNNN`); `frame_idx` is an int | — |
| Images | `images/Images_jpg/{video_id}/{segment_id}/{image_id}.jpg`, the same pattern as the official loader (line 78) | `IMAGE_PATH_TEMPLATE` |
| Videos | One clip per segment: `videos/Videos/{video_id}/{segment_id}.mp4` (158 files) | `locate_segment` (start 0, to be confirmed in 1G) |
| Info | `VideoID → {video_info: {title, url, type, location, weather}, splits_info: {SplitID: {front_view, side_view, back_view, start_time_stamp, end_time_stamp, delta_t}}}` | `info_for` |
| Official val split | Listed in the example loader (`val_splits`); metadata only | — |

**Audit result** (`$SKIER_RESULTS/step1/audit/`, run 2026-10-17):
- 16 videos, 158 segments, 1,982 images (`matches_expected: true`).
- **0 problems.** No shape, finite-value, range, visibility or `frame_idx` problems, and no missing or extra image files.
- Visibility values are only 0 (5,850) and 1 (41,718).
- Image sizes: 1,585 at 1920×1080 and 397 at 1280×720.
- Every inventory row has a race type and a view.

Reliable labels as a share of images (`visible = 1`), lowest first:

| Points | Reliable |
| --- | ---: |
| Ski tails R / L | 62% / 64% |
| Heels R / L | 73% / 75% |
| Toes, ski tips | 83–85% |
| Poles | 86–87% |
| Body joints | 90–99% |

Ski-tail and heel scores will rest on fewer points; always report the counts.

**Records to exclude** (rule: fewer than 2 reliable body + ski points, plan Step 2.2):
- `oKQFABiOTw8/27/oKQFABiOTw8_27_00020` (Giant Slalom, front view): 1 body point, 0 ski points.

## Findings that affect Step 2 (split)

**Race type is per video** (`video_info.type`):

| Type | Videos | Segments | Images |
| --- | ---: | ---: | ---: |
| Giant Slalom | 2 | 52 | 670 |
| Slalom | 2 | 32 | 539 |
| Combined Slalom | 1 | 18 | 395 |
| Super G | 2 | 26 | 152 |
| Follow Cam | 6 | 6 | 134 |
| Mixed | 1 | 18 | 62 |
| Downhill | 2 | 6 | 30 |

- **The EPFL page's per-sequence counts don't add up.** It lists 32 SL / 52 GS / 26 SG / 24 DH / 6 follow-cam = 140, not 158. The info file is used as the source of truth.
- **Follow-cam is 6 separate single-segment videos** (134 images), not 6 sequences of one video. Forcing all six into test would make them most of the test set's videos. A separate follow-cam evaluation set, reported on its own, is the likely better choice. **Decided: D11** (separate set).
- **Video sizes are very uneven.** Each of the largest five has 239–395 images (12–20% of the data), so a 15% test partition is about one large race video plus small ones. Spreading race types across partitions is limited: each type has only 1–2 videos.

Per video:

| video_id | Type | Segments | Images |
| --- | --- | ---: | ---: |
| he3w2n9WvrI | Combined Slalom | 18 | 395 |
| dpFlXMvvZ9E | Downhill | 3 | 14 |
| v7PADD2_I00 | Downhill | 3 | 16 |
| 37HtpPpdvp0 | Follow Cam | 1 | 33 |
| 3wkVqupcMAk | Follow Cam | 1 | 7 |
| FW1S3VAKui8 | Follow Cam | 1 | 29 |
| LjXeTB3Wl10 | Follow Cam | 1 | 44 |
| OeDt6D_9Rh4 | Follow Cam | 1 | 6 |
| nq2qj0fhYjc | Follow Cam | 1 | 15 |
| uLW74013Wp0 | Giant Slalom | 22 | 284 |
| oKQFABiOTw8 | Giant Slalom | 30 | 386 |
| NrWcP1s3QC0 | Mixed | 18 | 62 |
| 5UHRvqx1iuQ | Slalom | 11 | 300 |
| qxfgw1Kd98A | Slalom | 21 | 239 |
| zW1bF2PsB0M | Super G | 18 | 114 |
| SVYdH614G_M | Super G | 8 | 38 |

**Views are per segment** (images): front 676, front+side 743, front+side+back 232, side+back 140, side 101, back 90. Several flags can be true for one segment.

## 1F. Overlay review

**Done 2026-10-24 (Ruichao): all 30 overlays pass all six checks; no problems recorded.**

- Drawn on the laptop with `draw_gt_overlays.py --n 30 --seed 230`, pilot videos only: 8 `5UHRvqx1iuQ`, 15 `oKQFABiOTw8`, 7 `zW1bF2PsB0M`. The IDs are in `$SKIER_RESULTS/step1/overlays/overlay_ids.txt`.
- **Check 6 refers to visible points only.** The reference box uses reliable (`visible = 1`) body and ski points (`geometry.reference_box`), so hidden guesses can lie outside it. Example: `oKQFABiOTw8/24/oKQFABiOTw8_24_00016`, where a TV graphic covers both ankles and all 8 ski points (`vis = 0`), so the box ends at the knees.
- **How common that is, over all 799 usable pilot images:** 162 have a hidden body or ski point outside the padded (×1.25) crop, 161 of them involving a ski point. A visible point is never outside it, by construction. Hidden points are masked out of the loss and never scored, so the crop drops nothing that is trained or measured.

## 1G. Video-frame alignment

**Done 2026-10-26 (laptop; pilot segments only). Labels and clips align exactly, with no offset.**

| Segment | Clip | Labelled images | Best match | Offset | Verified |
| --- | --- | ---: | --- | --- | ---: |
| `5UHRvqx1iuQ/0` | 1920×1080, 25 fps, 521 frames (20.84 s) | 63 (`frame_idx` 4–520) | NCC 0.9999–1.0 | 0 for all | 54 |
| `oKQFABiOTw8/22` | 1920×1080, 25 fps, 384 frames (15.36 s) | 31 (`frame_idx` 6–381) | NCC 0.9999–1.0 | 0 for all | 27 |

- **Index convention:** `frame_idx` is the **0-based decode index within the segment's own clip** (`videos/Videos/{video_id}/{segment_id}.mp4`). `locate_segment` returns start 0, which is correct.
- **The labelled images are frames of these clips:** every best match scores 0.9999–1.0, and the clip lengths agree with the info file's `delta_t` (20.80 s, 15.34 s).
- **Unverified rows (13 of 94):** they fail only the margin rule (`MIN_MARGIN` 0.02). In near-static stretches, such as the start gate, neighbouring frames score almost as high, yet the best match is still the exact frame. `MIN_SCORE` is set to 0.99 from these results.
- Evidence: `results/step1/alignment/`. A `gt_clip.mp4` for each segment is in `$SKIER_RESULTS/step1/alignment/<video>__<segment>/`. Watching it is optional, given exact pixel matches.
- **Effect:** Step 7 can place labelled frames on the video timeline exactly, so the temporal comparison isn't a limitation.
