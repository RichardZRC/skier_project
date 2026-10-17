# Step 1 guide: prove the skier project is ready to implement

**Owner:** Ruichao Zhang — ruichaoz@stanford.edu  
**Planning date:** October 17, 2026  
**Time budget:** approximately 12 active hours  
**Full plan:** [Six-week implementation plan](skier_implementation_plan.md)

## What Step 1 must accomplish

Before training, answer four questions with files you can inspect:

1. Can I obtain the images, labels, and videos needed for the core project?
2. Do I understand exactly what each label and coordinate means?
3. Can I associate a labeled image with the correct video frame?
4. Can my chosen environment run pretrained pose inference and a short tracker example?

The result is a small evidence package and a clear decision to proceed, proceed with a limitation, or resolve a blocker. This step does not include a full model-training run or a polished application.

**Preserve the agreed scope:** core tracking, centering, and 2D body/ski landmarks; amateur follow-camera footage first among stretch goals; 3D and coaching at lower priority.

The commands below are instructions for your development machine. The project-specific script names are proposed files to create. They have not been installed or executed against the real datasets during this planning task. Code blocks are starter implementations or API examples, with placeholders explicitly identified.

## Session plan

| Task | Active time | Evidence produced |
| --- | ---: | --- |
| 1A. Create workspace and scope record | 0.5 h | Repository structure and scope |
| 1B. Obtain data and record provenance | 1.5 h | Download/access ledger |
| 1C. Select and verify environment | 2 h | Hardware/software record |
| 1D. Inspect annotation structure | 1.5 h | Schema and count report |
| 1E. Draw ground-truth overlays | 2 h | Audited image gallery |
| 1F. Verify video-frame alignment | 1.5 h | Alignment table and examples |
| 1G. Run pretrained smoke tests | 2 h | Pose predictions; tracking if available |
| 1H. Record readiness decision | 1 h | Audit and next actions |
| **Total** | **12 h** | |

Download waits and access approval are separate. Start the request early; work on pose data while waiting. If environment troubleshooting consumes the time limit, record the problem and use the fallback in 1C instead of losing several days.

## 1A. Create a small, reproducible workspace

Run this in the directory where you want the project to live:

~~~bash
mkdir -p skier-project
cd skier-project
git init
mkdir -p docs configs scripts src tests
mkdir -p data/raw data/manifests data/processed
mkdir -p artifacts/pretrained artifacts/step1 results vendor
~~~

Create a Git ignore policy before downloading large assets:

~~~gitignore
data/raw/
data/processed/
artifacts/pretrained/
vendor/
*.pth
*.pt
*.mp4
__pycache__/
.venv/
~~~

Track small audit documents, configuration, manifests, and source code. Keep dataset archives and pretrained weights outside Git. Decide separately which small figures/results belong in version control.

Create `docs/scope.md` with:

~~~markdown
# Scope

Owner: Ruichao Zhang
Email: ruichaoz@stanford.edu
Budget: 6 weeks × 20 hours

Core:
- Track one skier initialized by a bounding box.
- Produce a smooth centered replay.
- Estimate and evaluate 2D body and ski landmarks.

Stretch order:
1. Amateur footage filmed by another moving skier.
2. Lower priority: exploratory 3D and preliminary coaching.

Primary learning question:
Does targeted augmentation improve ski-landmark accuracy and
robustness to imperfect crops relative to ordinary fine-tuning?

Not required for the core:
Automatic target selection, real-time deployment, multiple-target tracking,
broadcast camera-cut recovery, or a separate learned crop controller.
~~~

Start `docs/decisions.md`. Each entry should state the decision, reason, date, and effect on evaluation. A short record now prevents confusion when a dependency or dataset detail changes later.

## 1B. Obtain the minimum useful data

### Ski 2DPose: start here

Open the [official EPFL page](https://www.epfl.ch/labs/cvlab/data/ski-2dpose-dataset/). Download the labels, additional information, example loader, and one image archive. Add the source videos for alignment testing. The page lists JPEG images at about 352 MB and video at about 519 MB. [R1]

Choose JPEG initially for a straightforward inspection workflow. Keep the original archives and metadata unchanged; extract working copies separately. Do not download every image encoding.

Suggested local arrangement, adapted to the actual archive layout:

~~~text
data/raw/ski2dpose/
    archives/
    labels/
    images/
    metadata/
    videos/
    official_loader/
~~~

The folder arrangement above is your project convention, not a claim about archive internals. Inspect archive contents first. Preserve original relative paths in your manifest.

Record one row per downloaded item:

| Field | Example of what to record |
| --- | --- |
| Asset | Labels, JPEG images, metadata, videos, loader |
| Source | Official page and resolved download URL |
| Date | Actual download date |
| Local path | Actual relative path |
| Size | Actual bytes |
| Hash | SHA-256 of downloaded file |
| Access/license note | Exact statement available with the asset |
| Status | Downloaded, extracted, validated, or blocked |

Use an ordinary checksum command:

~~~bash
sha256sum path/to/downloaded_archive
~~~

The proposal noted uncertainty about some redistribution terms. Carry that uncertainty in the ledger; do not treat a statement about source videos as permission to redistribute every annotation or asset. This is a provenance task, not a reason to postpone local technical inspection.

### SkiTB: start the access process immediately

Use the [official toolkit](https://github.com/matteo-dunnhofer/SkiTB-toolkit) and its linked access form. Request the alpine material needed for the project. The core needs annotated single-camera clips and split metadata; downloading all skiing disciplines is unnecessary. The repository identifies its code and annotations as CC BY-NC 4.0. [R2]

You should submit any access form yourself, with the actual information it requests. This guide does not submit a request or send anyone a message.

Record:

- Request date and current status.
- Actual available files and download instructions.
- Location of split JSON files.
- Any restrictions on sharing example video.
- A follow-up date if access is pending.

**Do not wait for SkiTB to begin the pose audit.**

### Pick development examples before model comparisons

Choose a small set of source-video IDs for the pilot, without using model scores. These videos must later remain in training/development. Save the list as `data/manifests/pilot_source_ids.txt`.

Use a few varied sequences rather than twenty nearly identical neighboring frames. Do not inspect every future test image manually while developing the method.

**Completion evidence:** assets or access status are documented, raw files are preserved, and pilot sources are identified.

## 1C. Select an environment and verify it

### Separate data inspection from GPU training

Data parsing and overlays can run on CPU. For training, use a single available CUDA GPU as the initial target. Roughly 8–16 GB VRAM is a planning starting point for this model/resolution, not a memory guarantee. Measure the actual footprint later.

A Linux GPU environment is a practical default. You can edit on another machine. If your available hardware differs, preserve the same data contracts and measure feasibility before changing the model.

Record:

- OS and Python version.
- GPU model, VRAM, driver, and availability.
- RAM and free disk.
- PyTorch version and its compiled CUDA version.
- MMPose, MMCV, MMEngine, NumPy, and OpenCV versions.
- Any session duration limits or storage persistence restrictions.

Useful commands:

~~~bash
python --version
nvidia-smi
df -h .
~~~

If no NVIDIA GPU is attached, `nvidia-smi` failing is expected; document where training will run.

### Use a compatible stack, then freeze it

Use **MMPose v1.3.2 as a reproducible starting revision**, matching the code inspected for this plan. This is not a claim that it is the newest release. Read its [installation instructions](https://mmpose.readthedocs.io/en/latest/installation.html) and [version guidance](https://mmpose.readthedocs.io/en/latest/faq.html). The PyTorch/CUDA/MMCV combination matters because MMCV includes compiled extensions. [R3]

Recommended installation order:

1. Create an isolated Python environment; Python 3.10 is a conservative starting choice.
2. Select a PyTorch build compatible with your driver **and an available MMCV wheel**.
3. Install the corresponding MMCV and MMEngine versions.
4. Install the pinned MMPose source.
5. Run a real inference check.
6. Save the exact working versions.

Do not blindly combine the newest version of every package. The precise GPU install command depends on hardware not yet specified, so this guide intentionally does not invent a universally working CUDA command.

For example, if using Conda:

~~~bash
conda create -n skier-pose python=3.10 pip -y
conda activate skier-pose
python -m pip install pillow matplotlib pyyaml
~~~

Install PyTorch and matching OpenMMLab dependencies using their compatible instructions, then pin the MMPose source:

~~~bash
git clone --branch v1.3.2 --depth 1 https://github.com/open-mmlab/mmpose.git vendor/mmpose
python -m pip install -e vendor/mmpose
python -m pip check
python -m pip freeze > artifacts/step1/environment-pip-freeze.txt
git -C vendor/mmpose rev-parse HEAD > artifacts/step1/mmpose-commit.txt
~~~

Do not install MMDetection just to run top-down pose on a box you already have.

Check imports and the actual runtime:

~~~python
import importlib.metadata as metadata
import torch

for name in ("torch", "torchvision", "mmcv", "mmengine", "mmpose", "numpy"):
    print(name, metadata.version(name))
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM bytes:", torch.cuda.get_device_properties(0).total_memory)
~~~

A successful import is necessary but insufficient: Step 1G runs the model.

**Troubleshooting limit:** after approximately 2–3 focused hours without inference, prefer a known compatible course environment. If none is available, evaluate a thin PyTorch implementation around [official HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch), with the same input/output and metric contracts. That fallback still requires work; do not treat it as an instant replacement.

A separate tracker environment is acceptable if old tracking dependencies conflict with the pose stack. Exchange predictions through files rather than forcing incompatible packages together.

## 1D. Inspect the exact annotation schema

### Keypoint order

Create `configs/ski24_schema.json` using the following zero-based order. The names/order come from the official annotation definition. [R1]

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

The resulting project groups are **14 body points, 8 ski points, and 2 pole points**. Model all 24 to preserve the schema; score body and skis separately. Do not silently reduce the ski representation to tips/tails.

“Right” refers to the skier’s side, not the right side of the image. When skis cross, retain the anatomical assignment. A 2D skeleton is just these points and drawing connections; no 3D body mesh is required.

### Coordinates, visibility, and frame identifiers

The labels are nested by video, segment, and image. Each image includes an `annotation` array and `frame_idx`. Coordinates are normalized by image width/height. The visibility flag separates reliable labels from occluded estimates; frame indices refer to the corresponding video segment. [R1]

Your conversion is:

~~~python
xy_pixels = annotation[:, :2] * [image_width, image_height]
reliable = annotation[:, 2] == 1
~~~

Keep occluded estimates in the raw data, but exclude them from primary supervision/evaluation. A coordinate value of zero is not by itself proof of a missing label.

Do not silently clamp invalid points. Count them, inspect examples, and document whether they are out-of-frame, uncertain, or corrupted.

### A small annotation-inspection script

Save this starter code as `scripts/inspect_pose_labels.py`. It expects the documented nested JSON structure and intentionally fails on unexpected structures.

~~~python
import argparse
import collections
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("labels", type=Path)
args = parser.parse_args()

raw = json.loads(args.labels.read_text())
per_video = collections.Counter()
visible_per_joint = np.zeros(24, dtype=np.int64)
flag_counts = collections.Counter()
problems = []
segment_count = 0

for video_id, segments in raw.items():
    segment_count += len(segments)
    for segment_id, images in segments.items():
        for image_id, record in images.items():
            key = f"{video_id}/{segment_id}/{image_id}"
            points = np.asarray(record["annotation"], dtype=float)
            if points.shape != (24, 3):
                problems.append([key, "wrong_shape", list(points.shape)])
                continue
            per_video[video_id] += 1
            if not np.isfinite(points).all():
                problems.append([key, "nonfinite"])
            flags = points[:, 2]
            flag_counts.update(str(x) for x in flags)
            visible_per_joint += flags == 1
            outside = ((points[:, :2] < 0) |
                       (points[:, :2] > 1)).any(axis=1)
            if outside.any():
                problems.append([key, "outside_unit_interval",
                                 np.flatnonzero(outside).tolist()])
            if "frame_idx" not in record:
                problems.append([key, "missing_frame_idx"])

print(json.dumps({
    "video_count": len(raw),
    "segment_count": segment_count,
    "valid_shape_image_count": sum(per_video.values()),
    "images_per_video": dict(per_video),
    "reliable_counts_by_joint": visible_per_joint.tolist(),
    "visibility_values": dict(flag_counts),
    "problems": problems,
}, indent=2))
~~~

Run it with the actual extracted labels path:

~~~bash
python scripts/inspect_pose_labels.py path/to/actual_labels.json > artifacts/step1/label_summary.json
~~~

This script audits annotation structure only. Add a second pass that resolves every record to an actual image, reads its dimensions, and records:

- unreadable or missing files;
- labels without images and images without labels;
- duplicate sample IDs;
- repeated frame indices within a segment;
- actual image sizes;
- reliable body/ski/pole counts;
- a stable path linking each sample back to its raw record.

Use a mapping based on the official loader and observed paths. Do not guess that `ImageID` is a complete filename or that directory ordering corresponds to frame ordering.

Save the resulting rows as `data/manifests/pose_inventory.csv`. Draft schema fields now; Step 2 finalizes the full adapter and splits.

**Completion evidence:** counts are measured from real data, each pilot label resolves to its image, and the label order is documented.

## 1E. Draw and inspect ground truth before using a model

### Build a small diagnostic gallery

Create approximately 24–40 overlays from the designated development sources. Cover large/small subjects, several views, crossed skis, and some occlusion.

For each image:

1. Read its original width and height.
2. Convert normalized coordinates using those dimensions.
3. Draw reliable points as filled markers.
4. Draw occluded estimated points as hollow markers, visibly distinct.
5. Label each point with its zero-based index.
6. Use different colors for body, skis, and poles.
7. Draw a reference box around reliable body/ski points.
8. Save the original-resolution overlay and a contact sheet.

Initially label points by index, even if the result looks crowded. It exposes ordering mistakes much faster than an unlabeled skeleton.

Suggested drawing connections are a project visualization convention:

- Body: head–neck; neck–shoulders; shoulder–elbow–hand; shoulder–hip; hips linked; hip–knee–ankle.
- Each ski: tip–toes–heel–tail.
- Optional: hand–pole basket.
- Draw an edge only when its endpoint policy is explicit. Use dashed or omitted edges for unreliable endpoints.

Do not draw helper joints as though they were additional ground-truth labels.

### What to inspect manually

| Observation | Likely explanation to check |
| --- | --- |
| All points are compressed into the top-left corner | Normalized coordinates were not converted |
| X/Y placement looks swapped | Width/height or coordinate order was reversed |
| Body looks correct but equipment is nonsensical | Keypoint order or group mapping is wrong |
| Front and rear views appear to swap sides | Image-side and anatomical-side semantics were confused |
| Points are consistently ahead of the skier | Wrong video frame or coordinate transform |
| Only occluded points look implausible | They may be uncertain estimates; inspect the visibility policy |
| Tips vanish after cropping | The crop definition is too tight or based only on body joints |

Record findings in `docs/data_audit.md`. Do not “correct” the public labels merely because the model disagrees with them. If genuine annotation defects are found, document objective inclusion/exclusion criteria before fitting the final system.

### Avoid a visibility-conversion trap

If you later export COCO-style annotations:

- Preserve the original points/flags in a canonical sidecar.
- Either produce a reliable-only training view, with unreliable points masked out, or implement an explicit target mask.
- Do not assume COCO visibility 1 means “ignore during training.” The inspected MMPose parser uses both positive codes as supervised. [R4]

For the first project implementation, a reliable-only training view is the easier policy to audit. Synthetic occlusion is a later augmentation decision and should not rewrite the raw reliability flags.

**Completion evidence:** you can point to several overlays and explain every marker, side assignment, and omitted label.

## 1F. Verify video-frame alignment

This task enables a later comparison of reference-crop pose accuracy with tracker-crop pose accuracy, without annotating every video frame.

For one or two pilot segments:

1. Locate the video associated with the segment.
2. Read its dimensions, duration, FPS, and frame count.
3. Take an annotated image from near the beginning, middle, and end.
4. Decode candidate frames around the documented index.
5. Compare the video frame with the provided image and overlay the labels.
6. Resolve zero-based versus one-based indexing and any segment offset.
7. Repeat the check across the segment; a match at only one timestamp is insufficient.

Prefer sequential decoding for the diagnostic. Random seeking can introduce decoder/codec ambiguity. Allow for JPEG/video compression differences; exact pixel equality is not required. The athlete’s silhouette, ski positions, and scene should align.

If the archive contains a full source video rather than the expected segment, use the supplied metadata to establish offsets. Do not assume “frame_idx / FPS” is an absolute timestamp in every file.

Create `artifacts/step1/alignment.csv`:

~~~text
source_video_id,segment_id,image_id,label_frame_idx,video_file,
resolved_decode_index,offset_policy,verified,notes
~~~

Once verified, create a short clip with ground truth drawn **only at labeled frames**. Leave other frames without ground-truth markers. Interpolated labels are not additional ground truth.

**If alignment fails:** spend up to the allocated 1.5 hours on the pilot, record the exact mismatch, and continue with image-level pose work. The project can still train and evaluate pose, but true tracking-to-pose evaluation remains unresolved. Do not pretend that feeding sparse labeled images to a tracker is equivalent.

## 1G. Run small pretrained smoke tests

### Pose smoke test

Obtain the COCO-trained HRNet-W32 checkpoint corresponding to the inspected configuration. With an installed MMPose/MIM environment, the intended model lookup is:

~~~bash
mim download mmpose --config td-hm_hrnet-w32_8xb64-210e_coco-256x192 --dest artifacts/pretrained
~~~

Verify that the resolved config and checkpoint match. Record their filenames, source, and hash. If the model registry lookup fails, use the matching entry in the upstream model collection rather than inventing a checkpoint URL.

The following example runs a supplied box through the public API. Replace the three paths and four box values with actual local inputs. The box is `[x1, y1, x2, y2]` in original-image pixels. [R5]

~~~python
import numpy as np
import torch
from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules

register_all_modules()
config_path = "REPLACE_WITH_DOWNLOADED_CONFIG.py"
checkpoint_path = "REPLACE_WITH_MATCHING_CHECKPOINT.pth"
image_path = "REPLACE_WITH_PILOT_IMAGE.jpg"
box_xyxy = np.array([[100, 100, 700, 900]], dtype=np.float32)
# Replace this illustrative box with the actual pilot skier box.

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = init_model(config_path, checkpoint_path, device=device)
result = inference_topdown(
    model, image_path, bboxes=box_xyxy, bbox_format="xyxy"
)[0]
pred = result.pred_instances
points = pred.keypoints[0]
scores = pred.keypoint_scores[0]
assert points.shape == (17, 2)
assert scores.shape == (17,)
assert np.isfinite(points).all()
np.savez(
    "artifacts/step1/generic_pose_prediction.npz",
    keypoints=points, scores=scores
)
~~~

Use a box containing the actual skier. Passing the whole snowy frame as one “person crop” can make a valid model appear broken.

Draw the resulting 17-point COCO skeleton with its own metadata. Do not draw those outputs using the 24-point ski schema.

Expected outcome:

- One set of finite body predictions appears near the target.
- Difficult joints can be wrong.
- Ski tips/tails are absent because the generic model does not predict them.
- This verifies execution, image handling, and coordinate output; it is not a baseline score yet.

When timing GPU inference, warm up the model and synchronize CUDA before and after timing. Save actual timing, device, and whether decoding/visualization are included.

### Tracking smoke test, if access is available

Clone the official toolkit and the tracker implementation it references; record their revisions and checkpoint provenance. Start with one short **training/development single-camera clip**.

Inspect the downloaded example before running it. At the revision reviewed for this guide:

- The parser spells the argument `--split_file`.
- Single-camera mode is selected with `--mode SC`.
- The example’s data root may require configuration.
- One dataset-loader path uses `ST` while another uses `SC`. Resolve it against the actual archive.
- If NumPy rejects a newline passed as `loadtxt`’s delimiter, use its default whitespace parsing for one-number-per-line files, leaving comma-separated boxes unchanged. This is a conditional compatibility fix, not a reason to downgrade everything. [R6, R7]

A full validation command later has this shape; substitute the actual data-root setup and split path:

~~~bash
python example_skitb.py --discipline AL --mode SC --split_file /absolute/path/to/actual_athlete_split.json --split val
~~~

For Step 1, adapt the example to one training clip instead of running a complete benchmark.

Save:

- Original frames with predicted boxes.
- A box per processed frame.
- Initialization frame and rectangle.
- Confidence values, if supplied.
- Clip timestamps and any failure.
- Whether any reset occurred.

Inspect at least the first, middle, and last frames. Confirm that no ground-truth boxes are being fed back after initialization.

If SkiTB is pending, a verified Ski 2DPose video can support a qualitative tracker smoke test. Initialize it once using a development-frame box. Do not label this an official tracking benchmark.

**Completion evidence:** at least one real pose inference works; tracking either works on a short clip or has a concrete, bounded blocker.

## 1H. Make a readiness decision

Create `docs/step1_readiness.md` using this template:

~~~markdown
# Step 1 readiness

Date:
Active hours spent:
Owner: Ruichao Zhang
Email: ruichaoz@stanford.edu

## Data
- Pose images/labels available:
- Actual usable image count:
- Actual source-video count:
- Pilot sources reserved for training:
- Annotation issues:
- SkiTB access/status:
- Provenance/license uncertainties:

## Label understanding
- 24-point schema checked:
- Body/ski/pole group mapping checked:
- Normalized-to-pixel conversion checked:
- Visibility policy checked:
- Ground-truth overlays reviewed:

## Environment
- GPU/VRAM:
- Python/PyTorch/CUDA:
- MMPose/MMCV/MMEngine:
- Revisions/checkpoint hashes:
- Working pose inference:
- Working tracking inference:
- Timing/memory observations:

## Temporal alignment
- Segments checked:
- Successful alignments:
- Index/offset convention:
- Remaining blocker:

## Decision
- Ready / ready with limitation / blocked:
- Reason:
- Next action:
- Time limit for resolving blocker:
- Any scope change to document:
~~~

Decision rules:

| State | Meaning | Next action |
| --- | --- | --- |
| **Ready** | Labels/images align; pose inference works; tracking data and temporal route are usable | Proceed to Step 2: splits, geometry, and metrics |
| **Ready with limitation** | Pose work is feasible; tracking access or video alignment is pending | Proceed with pose; assign a deadline to the unresolved item |
| **Blocked** | Labels/images cannot be obtained/interpreted, or no viable model runtime is identified | Resolve the specific blocker before scheduling training |

By the end of Week 1, unresolved SkiTB access should trigger the main plan’s fallback. Lack of a tracker must not silently eliminate tracking evaluation from the report: report the narrower scope explicitly.

## End-of-step checklist

- [ ] Scope and stretch priorities are recorded.
- [ ] The access/download ledger contains actual status and provenance.
- [ ] Raw annotations remain unchanged.
- [ ] Pilot source IDs are reserved for development.
- [ ] All 24 names and flip pairs are correct.
- [ ] Visibility and coordinate conventions are understood.
- [ ] Annotation/image counts and unresolved files are recorded.
- [ ] Ground-truth overlays have been manually checked.
- [ ] Video alignment has been verified or explicitly marked unresolved.
- [ ] A real pretrained pose inference works.
- [ ] Tracker execution works or has a named blocker.
- [ ] Environment and checkpoint details are recorded.
- [ ] The readiness decision names the next action.

## Bounded tasks to give your AI coding assistant

Give one task at a time, with real local paths. Inspect its outputs before moving on.

**Task 1: annotation audit**

> Read the official labels, metadata, and example loader. Write an audit script that preserves raw data, inventories images, validates 24-point shapes and visibility values, and reports unmatched files. Preserve source-video and segment IDs. Do not train a model or randomly split frames.

**Task 2: ground-truth visualization**

> Using the audited schema, generate indexed overlays for my designated development samples. Separate reliable and uncertain points, use anatomical left/right labels, and draw body, ski, and pole groups distinctly. Save the sample IDs used. Do not substitute model predictions for labels.

**Task 3: frame alignment**

> For these two development segments, match annotated images to decoded video frames using the supplied metadata. Verify several timestamps, record the indexing/offset convention, and fail explicitly if no match can be established. Do not interpolate extra ground truth.

**Task 4: pretrained inference**

> Run the pinned COCO HRNet-W32 checkpoint on one supplied skier box, save original-image coordinates and scores, and render with the COCO schema. Record config, checkpoint hash, package versions, and device. Do not fine-tune or map unsupported joints.

These tasks produce inspectable evidence and keep Step 1 small enough to finish.

## References

Sources were checked October 17, 2026. Download success, hardware compatibility, and measured dataset counts still need to be established on your machine.

- **[R1]** [EPFL Ski 2DPose](https://www.epfl.ch/labs/cvlab/data/ski-2dpose-dataset/).
- **[R2]** [Official SkiTB toolkit and access instructions](https://github.com/matteo-dunnhofer/SkiTB-toolkit).
- **[R3]** [MMPose installation](https://mmpose.readthedocs.io/en/latest/installation.html), [compatibility FAQ](https://mmpose.readthedocs.io/en/latest/faq.html), and [PyTorch installation selector](https://pytorch.org/get-started/locally/).
- **[R4]** [MMPose v1.3.2 COCO-style dataset parser](https://github.com/open-mmlab/mmpose/blob/v1.3.2/mmpose/datasets/datasets/base/base_coco_style_dataset.py).
- **[R5]** [MMPose v1.3.2 inference API](https://github.com/open-mmlab/mmpose/blob/v1.3.2/mmpose/apis/inference.py) and [HRNet-W32 configuration](https://github.com/open-mmlab/mmpose/blob/v1.3.2/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py).
- **[R6]** [SkiTB example CLI](https://github.com/matteo-dunnhofer/SkiTB-toolkit/blob/main/example_skitb.py).
- **[R7]** [SkiTB dataset loader](https://github.com/matteo-dunnhofer/SkiTB-toolkit/blob/main/toolkit/datasets/skitb.py).

