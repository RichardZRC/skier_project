# Evaluation protocol

**Status: DRAFT (2026-10-17). Ruichao reviews it and freezes it together with `data/manifests/pose_split_v1.json`.** After the freeze, changes go through `docs/decisions.md` with a reason, and never because of a score. Code: `src/skier/eval/`, `src/skier/splits.py`.

## 1. Coordinates and labels

- **Pixels:** keypoints live in original-image pixels: `x = x_norm × W` and `y = y_norm × H`, using **each image's own size** (the dataset mixes 1920×1080 and 1280×720).
- **Pixel convention:** OpenCV's, with pixel centers at integer coordinates. The dataset doesn't say whether normalized 0 means a pixel's edge or its center, which leaves an ambiguity of at most 0.5 px. That is far below the smallest tolerance used: PCK@0.05 on a typical 300 px reference diagonal is 15 px.
- **Reliable label:** `visible == 1`, meaning visible or confidently labeled. `visible == 0` is an occluded best guess. It is **never** scored and never supervises the loss.
- **Reference box:** the tight, unpadded box around the reliable **body + ski** points (poles excluded). Its diagonal `d` is the PCK normalizer.
- **Exclusions:** a record with fewer than 2 reliable body+ski points, or with `d < 1 px`, has no reference box.
  - It is kept, flagged with `excluded_reason`, and listed in the split manifest. It is never silently dropped.
  - Currently 1 record: `oKQFABiOTw8/27/oKQFABiOTw8_27_00020` (a train video).

## 2. Pose metrics

- **PCK@t** for a group of landmarks is `hits / total`, where:
  - `total` = the reliable labels in the group;
  - `hit` = a finite prediction within `t × d` (`≤`) of the label.
- **Missing predictions:** a missing prediction (NaN) is a **miss** and stays in the denominator. Confidence thresholds are **never** applied inside metrics.
- **Thresholds:** `t = 0.1` is the headline; `t = 0.05` is also reported for skis.
- **Groups:** body (14) and ski (8) are reported separately; poles (2) are supplementary only.
- **Counts:** every table gives `hits / total` counts next to percentages, plus per-video rows next to the pooled number.
- **Also reported:** the mean normalized error `dist / d` over the reliable labels that have a prediction, plus the missing-prediction rate.
- **E0** (the unchanged COCO model) is scored only on the 10 joints both schemas share: shoulders, elbows, hips, knees, ankles.
  - COCO wrist → Ski hand goes in a separate, explicitly approximate row.
  - E0's ski PCK is **N/A**, not 0.
- **Comparisons between experiments** use identical sample IDs and denominators. Paired differences use a bootstrap over source videos, not frames (Step 8).

## 3. Split (D4, D11)

- **Whole source videos only:** a video's images never sit in two partitions.
- **Follow-cam (D11):** the 6 follow-cam videos (134 images) form a separate **`followcam`** evaluation set. They are never trained on, are outside the 70/15/15 split, and are reported as their own row.
- **Generator:** `make_pose_split` with targets 70/15/15 by image count over the main videos; 2–4 videos each in val and test; at least 400 reliable ski points per held-out partition.
  - Score: `|val − 0.15| + |test − 0.15| + 0.05 × race types missing` in each held-out partition. Race types exclude Follow Cam.
  - It searches exhaustively, and ties are broken by seed 230.
- **Pilot videos:** forced into train, so studying them closely never touches val or test.
- **The frozen manifest** records the video lists, sample IDs, counts, exclusions, generator arguments and a content SHA-256. The CLI refuses to overwrite it without `--force`.
- **Usage:** val is for every tuning and model-selection decision. **Test is evaluated once**, in Step 8, with frozen settings. The follow-cam set follows the same rule as test. Never re-split after seeing scores.

## 4. Tracking metrics (SkiTB AL, single-camera clips)

- **IoU:** averaged over **visible annotated** frames, excluding the initialization frame. A missing tracker box counts as IoU 0.
- **Means:** the pooled mean is over all such frames; the per-clip mean is the average of the clip means. Both are reported.
- **Success curve:** the fraction of frames with IoU above each threshold 0, 0.05, …, 1, and its AUC.
- **Failure episodes** (our diagnostic, not the toolkit's official metric):
  - IoU < 0.1 on visible frames for at least 0.5 s;
  - invisible frames neither break nor end an episode;
  - duration = (last − first failing frame + 1) / fps.
- **The toolkit's official metrics** are kept in separate, correctly named columns.
- **The tracker** is initialized once per clip, and ground truth is never fed back.

## 5. Replay metrics

Defined in Step 4 (plan §9). They will be added here before any replay tuning.
