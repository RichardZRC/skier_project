# Skier project: six-week technical implementation plan

**Owner:** Ruichao Zhang — ruichaoz@stanford.edu  
**Planning date:** October 17, 2026  
**Budget:** one person, six weeks, approximately 20 working hours/week  
**Companion:** [Detailed Step 1 guide](skier_step_1_guide.md)

This plan implements the existing proposal. It describes work to do; no dataset download, environment installation, training, or benchmark result is claimed here. Public documentation and selected source files were checked for planning. Hyperparameters and time allocations below are starting choices, not experimentally validated settings.

## 1. What you will build

**Core deliverable:** given a single-camera skiing video and a box selecting the skier in the first usable frame, produce a smooth centered replay with 2D body and ski landmarks, plus reproducible quantitative results.

The main learning contribution is **ski-specific pose fine-tuning and a controlled study of robustness augmentation**. Use a fixed pretrained tracker. Centering is a geometric video-processing component; it does not need its own learned model.

Priority order:

1. **Core:** target tracking, body/ski landmark estimation, centered replay, evaluation, report.
2. **First stretch goal:** transfer to amateur footage where another moving skier films the target.
3. **Lower-priority stretch goals:** exploratory 3D reconstruction and preliminary coaching feedback. Neither should displace the amateur extension or core evaluation.

Keep the interface simple: one command to process a clip, a first-frame box, an output video, and machine-readable predictions. A web application, automatic skier selection, multi-person tracking, broadcast camera-cut recovery, real-time deployment, and tracker training are not needed for the core project.

### System structure

~~~mermaid
flowchart TD
    A["Video and initial box"] --> B["Fixed skier tracker"]
    B --> C["Expanded pose crop"]
    C --> D["Fine-tuned 2D landmark model"]
    B --> E["Smooth replay crop"]
    D --> F["Coordinate mapping and overlay"]
    E --> F
~~~

Run pose on the original frame through its own crop transform. Apply the replay crop afterward. This separation lets you change replay smoothing without changing pose predictions.

The replay repositions and resizes available pixels. It cannot recover detail lost to motion blur or reveal occluded equipment.

## 2. Decisions to settle early

| Decision | Starting choice | Why |
| --- | --- | --- |
| Learned component | COCO-pretrained HRNet-W32, adapted to 24 outputs | Matches the proposal and isolates the learning work |
| Tracker | Pretrained SiamFC through a thin SkiTB adapter | Avoids a second training project |
| Model input | Height 256 × width 192 | Small enough for an initial single-GPU experiment |
| Model output | 24 heatmaps; decoded image coordinates and scores | Retains the dataset schema |
| Main landmark groups | Body and skis reported separately; poles secondary | Equipment accuracy must not be hidden by easier joints |
| Main study | Ordinary fine-tuning versus targeted augmentation | A feasible, testable research question |
| Pose split | Whole source videos, approximately 70/15/15% by image count | Nearby frames must not leak across partitions |
| Tracking split | Official athlete-disjoint alpine split, single-camera mode | Keeps comparison interpretable |
| Evaluation mode | Offline processing | No need to optimize latency before accuracy |
| Manual annotation | None required initially; a small bounded fallback only | Preserve the six-week budget |

Do not interpret “professional data” as a precise dataset category. Ski 2DPose contains mixed skiing levels and some follow-camera sequences. That helps feasibility but does not establish generalization to arbitrary amateur footage. [S1]

### Data roles and limitations

| Data | What it supplies | What you can measure |
| --- | --- | --- |
| Ski 2DPose images/labels | 1,982 labeled images from 16 source videos; body/equipment landmarks | Pose accuracy and landmark containment in crops |
| Ski 2DPose associated video | Temporal context around annotated frames, if alignment is verified | Tracking-to-pose error propagation at labeled timestamps |
| SkiTB alpine subset | 100 parent videos, boxes, visibility, camera segments and splits | Tracking overlap and box containment |
| Optional independent amateur clips | Video; usually no labels | Qualitative transfer initially; quantitative results after annotation |

SkiTB does not supply your 24-point pose ground truth. Ski 2DPose landmarks do not automatically constitute official tracking boxes. Keep these distinctions in every result table. [S1, S2]

## 3. Schedule and dependencies

Hours mean your active work, including debugging and inspecting results. GPU execution and access-request waiting can run in parallel; neither is guaranteed to finish inside these estimates.

| Step | Work | Active hours | Scheduled |
| --- | --- | ---: | --- |
| 1 | Establish data and compute feasibility | 12 | Week 1 |
| 2 | Freeze splits, geometry, and metrics | 8 | Week 1 |
| 3 | Measure unchanged pose/tracking baselines | 12 | Week 2 |
| 4 | Build the initial centered replay | 8 | Week 2 |
| 5 | Train the first ski landmark model | 20 | Week 3 |
| 6 | Run the bounded augmentation study | 20 | Week 4 |
| 7 | Integrate and measure error propagation | 16 | Week 5 |
| 8 | Final evaluation, report, and demonstration | 16 | 4 hours in Week 5; 12 in Week 6 |
| 9 | Reserved time: debugging first, stretches second | 8 | Held until needed |
| **Total** | | **120** | |

The critical path is **usable pose labels → correct coordinate transforms and metrics → baseline → fine-tuning → controlled comparison → final report**. Tracking integration can proceed while training runs. Amateur data collection must never block this path.

Week-end checkpoints:

- **Week 1:** genuine data overlays, a working inference environment, frozen evaluation design.
- **Week 2:** baseline numbers and a first centered replay.
- **Week 3:** a trained model producing all 24 landmarks.
- **Week 4:** a defensible augmentation comparison.
- **Week 5:** integrated videos and quantified crop-induced pose errors.
- **Week 6:** final tables, failure analysis, reproducible code, report, and demo.

## 4. Step 1 — Establish feasibility before building the pipeline

**Budget:** 12 hours. The companion guide expands each task, including schema, commands, checks, and fallback decisions.

1. Create the repository and write down scope, hardware, and deliverables.
2. Start the SkiTB access process; obtain Ski 2DPose labels, images, metadata, example loader, and a small video sample.
3. Inspect actual files. Record counts, coordinate conventions, label ordering, and visibility.
4. Draw ground-truth landmarks on original images. Verify source-video/frame alignment on a short clip.
5. Run one pretrained pose inference; run a short tracking smoke test if access is available.
6. Produce a one-page feasibility decision with remaining blockers and dated next actions.

**Exit condition:** you have evidence that images and labels are usable, can run an existing model, and know how the two datasets will support evaluation. A working demo on every video is not required yet.

**Do not begin a long training job during this step.** A misplaced coordinate transform can yield decreasing loss and still invalidate the entire experiment.

## 5. Step 2 — Freeze the experimental contract

**Budget:** 8 hours.  
**Depends on:** Step 1 data audit.

### 2.1 Freeze splits before inspecting model results

For Ski 2DPose:

- Group by original source-video ID; keep every camera segment and extracted frame from a source together.
- Aim for approximately 70/15/15% of images. With only 16 source videos, exact percentages are not a priority.
- Choose a deterministic allocation using metadata and sample counts, before running model comparisons. Aim for at least two source videos in each held-out partition if feasible.
- Keep any pilot images used for debugging in training. Reserve other videos for validation and test.
- Inspect whether athlete identity links multiple source videos. If reliable identity metadata exists, merge linked groups. Otherwise describe the split as video-disjoint, not athlete-disjoint.
- Retain the original published split as metadata; this project split is a separate protocol and is not directly comparable with a published validation score.
- Save video IDs, image IDs, seed, group counts, frame counts, visible-label counts, and a manifest hash.

For SkiTB:

- Use the provided alpine athlete-disjoint train/validation/test membership.
- Derive single-camera clips from their parent video’s partition.
- Do not assign camera segments from one performance independently.
- Use training clips for development, validation for crop/controller settings, and test once the settings are frozen.
- Verify the actual JSON keys and counts; do not infer percentages from a filename.

If recordings overlap between datasets, place matching content in compatible partitions or exclude it from evaluation. Record this check even if no overlap is found.

### 2.2 Define a canonical record

Maintain an internal record with:

~~~text
sample_id, dataset, source_video_id, segment_id
frame_index, timestamp_seconds, image_path, image_width, image_height
keypoints_xy[24, 2], label_reliable[24]
reference_pose_box_xywh, tracking_box_xywh_if_available
split, annotation_source
~~~

Use pixel coordinates in the original image internally. Preserve the original annotations separately.

Distinguish three boxes:

1. **Tracking box:** the benchmark’s annotated or predicted target rectangle.
2. **Reference pose box:** an enclosure of reliable body/ski labels, used for pose experiments.
3. **Replay crop:** the region displayed in the output video.

Do not overwrite one with another; their definitions and evaluation roles differ.

### 2.3 Freeze pose geometry and preprocessing

Use a reference pose box enclosing reliable body and ski landmarks, excluding poles. Expand it initially by a factor of 1.25 about its center, then adjust to the model’s width/height ratio of 192/256 without stretching the skier. The 1.25 factor is a starting value; choose any revision on validation data.

If the crop goes outside the image, pad it and retain the full transform. Store the affine matrix and its inverse for every sample. Convert decoded predictions back into original-frame pixels before measuring errors.

Apply crop expansion exactly once. If your adapter already expands the box, remove or neutralize the framework's additional bounding-box padding; otherwise the actual crop will differ from the documented factor. Verify the resulting box numerically and on an overlay.

Specify conventions explicitly:

- Image arrays use height/width; MMPose configuration sizes use width/height.
- Source normalized coordinates become `x_px = x_norm × W` and `y_px = y_norm × H`.
- Use a consistent continuous coordinate convention for box math.
- Let the selected codec own image-to-heatmap coordinate conversion; avoid independently rounding coordinates.
- Match the pretrained checkpoint’s normalization and channel order. Do not normalize twice.
- Retain reliable labels in the evaluation denominator even when a bad predicted crop cuts them out.

### 2.4 Implement metrics independently of visualization

For image `i`, let `d_i` be the diagonal of the **unpadded reference pose box**, built from reliable body/ski labels. It stays fixed for all methods and crop variants on that image. Define:

\[
e_{ij}=\frac{\|\widehat p_{ij}-p_{ij}\|_2}{d_i},\qquad
\operatorname{PCK}_{G}@0.1
=100\frac{\sum_{i,j\in G}v_{ij}\mathbf{1}[e_{ij}\le0.1]}
{\sum_{i,j\in G}v_{ij}}.
\]

Here `G` is the body or ski group; `v` is the ground-truth reliability mask. Flag images with insufficient points or degenerate boxes, and report exclusions by reason.

**Primary:** ski PCK@0.1, pooled across reliable ski labels.  
**Secondary:** body PCK@0.1, both groups at 0.05, per-landmark PCK, per-source-video PCK, and normalized mean error on available predictions with missing-prediction rate reported alongside it.

This specifies the proposal’s normalization more precisely. Long skis can make the tolerance generous; the stricter 0.05 result and per-joint errors help interpret it.

Evaluation rules:

- Every reliable labeled point must either produce a prediction or count as incorrect for PCK.
- A confidence threshold must not silently remove difficult examples from accuracy metrics.
- Save raw predictions before confidence filtering or temporal smoothing.
- Pole scores are supplementary and never included in the main body/ski average.
- Do not reuse COCO AP/OKS with 17-joint defaults for this 24-point task. Use a custom metric or a separate evaluator.
- Report denominators, not only percentages.

### 2.5 Check the high-risk logic

Implement a few targeted checks:

- A point transformed into the crop and back returns within 0.5 original-image pixels.
- Horizontal flip applied twice restores coordinates and labels.
- A prediction displaced by `0.09d` passes PCK@0.1; one displaced by `0.11d` fails.
- Identical boxes have IoU 1; disjoint boxes have IoU 0.
- Source-video groups have no partition intersection.
- Missing or low-confidence predictions cannot inflate PCK.

**Outputs:** split manifests, geometry utilities, dataset adapter, metric specification, and these small correctness checks.

**Exit condition:** a deliberately perfect predictor scores 100% PCK, and deliberately incorrect points score as expected.

## 6. Step 3 — Establish baselines before changing the model

**Budget:** 12 hours.

### 3.1 Generic pose baseline

Run unchanged COCO-pretrained HRNet-W32 on the same reference crops used by later models. For the cleanest comparison, score the ten unambiguous corresponding joints: left/right shoulder, elbow, hip, knee, and ankle.

Do not equate COCO nose with dataset head, invent neck ground truth, or assume “hand” is identical to “wrist.” If inspection justifies a wrist/hand comparison, make it a separate, explicitly approximate result.

Save predictions and inference timing. The generic model has no ski outputs: show **N/A** for ski PCK, not zero.

### 3.2 Tracker baseline

Use pretrained SiamFC, initialized only at the first usable frame of each single-camera clip. Keep weights fixed throughout the project. Record checkpoint provenance and ensure any ski-specific checkpoint has not been trained on your held-out footage.

Evaluate:

\[
\operatorname{IoU}_t=\frac{|B_t^{pred}\cap B_t^{gt}|}{|B_t^{pred}\cup B_t^{gt}|}.
\]

Primary tracking score: mean IoU over annotated frames marked visible, excluding the initialization frame. Report both pooled-frame and per-clip averages. Also report IoU success curves, low-overlap episodes, and available processing speed.

Use an operational failure flag such as IoU below 0.1 for at least 0.5 seconds on visible evaluation frames. This is your diagnostic definition, not a claim about the benchmark’s official failure metric. Do not secretly reset the tracker using later ground truth.

Keep official toolkit metrics in separate columns with their actual names and protocols. The toolkit’s long-term confidence/F-score analysis is not interchangeable with your mean IoU.

**Source-code checks:** the current example uses `--split_file`, while README examples use a different spelling. The dataset loader also appears to enumerate `ST` in one single-camera path and read `SC` in another. Inspect the downloaded revision and folder structure, then apply the smallest documented fix if needed. These are source-inspection findings, not locally reproduced failures. [S6, S7, S8]

### 3.3 Inspect errors before fine-tuning

Create a gallery of roughly 30–50 training/validation examples covering:

- large versus tiny skier;
- side, frontal, and rear views;
- ski crossing and occlusion;
- blur, snow spray, and frame-edge truncation.

Use metadata where available. Do not create subjective test difficulty labels after seeing model outputs.

**Outputs:** baseline prediction files, initial tables, an error gallery, and measured timing/memory.

**Exit condition:** the baseline can be rerun from a saved command/config and produces the same sample IDs and metric definitions as subsequent experiments.

## 7. Step 4 — Build a centered replay without waiting for training

**Budget:** 8 hours.

### 4.1 Establish a simple reference implementation

Start with the tracker box, expand it to preserve ski context, and choose a fixed output aspect ratio. Use 16:9 as the initial replay format; keep the pose crop’s 3:4 ratio separate.

Represent desired crop state as `(cx, cy, log_width, log_height)`. Smooth its position and size:

\[
s_t=\alpha_t q_t+(1-\alpha_t)s_{t-1},
\qquad \alpha_t=1-\exp(-\Delta t/\tau).
\]

Start with `tau = 0.2 seconds`, then compare a small validation grid such as 0, 0.1, 0.2, and 0.4 seconds. Larger values smooth more and lag more. Preserve output FPS and timestamps.

First implement the box-based crop. Later, reliable ski predictions may expand its extent, but evaluate that as a separate controller variant.

### 4.2 Handle containment and uncertainty

- Enforce a minimum crop size and a maximum zoom.
- Expand quickly when the predicted target approaches the boundary; contract more slowly.
- Clamp/pad consistently at image borders.
- If tracking confidence becomes unreliable, briefly hold the last useful crop and zoom outward; do not present a stale pose as current evidence.
- Keep raw track output even when the renderer uses a fallback crop.
- A manual restart may be acceptable in a demonstration, but log it and exclude it from the no-reset tracking benchmark.

Initially treat tracker response scores as uncalibrated indicators. Tune thresholds on validation clips only.

### 4.3 Measure replay quality as a trade-off

| Metric | Ground truth | Interpretation |
| --- | --- | --- |
| Box containment | SkiTB target box | Fraction of target box inside crop |
| Landmark containment | Reliable body/ski points | Fraction inside crop; ski endpoints separately |
| Centering error | Ground-truth target center | Distance from crop center, normalized by crop diagonal |
| Crop acceleration | Crop-center trajectory | Frame-rate-aware change in crop velocity |
| Zoom behavior | Crop sizes | Area fraction and speed of log-scale changes |

For constant frame spacing, use:

\[
J=\operatorname{mean}_t
\frac{\|c_{t+1}-2c_t+c_{t-1}\|_2}
{D_{frame}\Delta t^2}.
\]

For variable frame spacing, compute velocity and acceleration using timestamps. Measure only within continuous segments. Crop acceleration is a jitter proxy, not a ground-truth measure of “correct camera motion.”

A static full-frame crop can score well on containment and smoothness. Therefore report crop area and centering alongside those scores. Compare controllers at similar zoom/containment or show a small trade-off table.

Provisional engineering target: at least 95% of visible ski endpoints retained on the selected labeled evaluation frames, with reduced crop acceleration relative to unsmoothed cropping at comparable zoom. This is a validation target to investigate, not a promised result.

**Outputs:** a reproducible centered video and raw-versus-smoothed crop comparison.

## 8. Step 5 — Train the ski-specific model

**Budget:** 20 active hours plus measured GPU runtime.

### 5.1 Adapt HRNet correctly

Use the MMPose HRNet-W32 heatmap configuration as a starting point, not as an unchanged training recipe. The inspected configuration uses `input_size=(192,256)`, `heatmap_size=(48,64)`, and a 17-channel head. [S3]

Required changes:

- Replace the final prediction layer with 24 channels.
- Load the full **COCO pose checkpoint** into matching weights. The backbone’s default initialization URL alone is not equivalent to a fully pose-pretrained model.
- Exclude incompatible output-layer tensors before loading. In PyTorch, `strict=False` does not by itself solve every tensor-shape mismatch.
- For the first experiment, randomly initialize the whole new final layer; keep that policy fixed across experiments.
- Replace dataset paths/metainfo, training length, evaluator, checkpoint-selection metric, and scheduler.
- Remove COCO person-detection result files.
- Disable half-body augmentation; it can intentionally discard the skis.
- Disable test-time flip augmentation initially for all models.
- Disable automatic learning-rate scaling unless explicitly configured for the actual batch size.

Use a custom 24-point metainfo file and COCO-style adapter. MMPose supports this route, including left/right swap definitions. [S4]

### 5.2 Supervise only reliable targets

For image `i` and landmark `j`, generate a Gaussian heatmap at the transformed labeled position. Start with sigma 2 heatmap pixels.

\[
\mathcal L =
\frac{\sum_{i,j}m_{ij}
  \left(\frac{1}{HW}\sum_{u,v}
  (\widehat H_{ijuv}-H_{ijuv})^2\right)}
{\max(1,\sum_{i,j}m_{ij})}.
\]

Use `m=1` only for reliable labels that are inside the augmented training crop. Skip all-invalid training samples.

Preserve raw visibility annotations. The default COCO-style parser in the inspected MMPose revision treats both positive visibility codes as supervised; converting uncertain points to COCO `v=1` would therefore not automatically ignore them. Export a documented reliable-only training view or pass an explicit mask. [S5]

This training mask differs from evaluation: cropped-out reliable evaluation points still count as misses.

Check the loss reduction as well as its mask. A stock weighted MSE may average over all channels rather than divide by the number of valid targets. Implement the equation above explicitly, or document the actual reduction and use it consistently across every comparison.

### 5.3 Debug by overfitting a tiny subset

Before full training:

1. Select 16–32 varied training images.
2. Disable random augmentation.
3. Train until predicted points fit these images closely.
4. Confirm body AND ski heatmaps peak near labels.
5. Inspect loss and decoded overlays together.

A very high training PCK is a debugging target, not a generalization result. If this fails, inspect transforms, heatmap scale, visibility masks, checkpoint loading, and decoder coordinates before changing architecture.

### 5.4 Run the first real fine-tuning experiment

Suggested initial recipe:

| Setting | Initial choice |
| --- | --- |
| Batch size | 16; reduce to 8 if necessary |
| Optimizer | Adam |
| Warm start | 2–3 epochs with backbone frozen, head LR 1e-3 |
| Fine-tuning | Up to 50 epochs; backbone LR 1e-4, head LR 5e-4 |
| LR schedule | Short warm-up if needed; reduce at roughly 60% and 85% of the chosen run |
| Validation | Every epoch |
| Checkpoint choice | Highest validation ski PCK@0.1 |
| Standard augmentation | Horizontal flip 0.5, modest rotation/scale |
| Seeds | One during development; more for finalists if budget permits |

Treat these as conservative initial settings. Use the same schedule and initialization policy in controlled comparisons. If small batches make batch-normalization unstable, freeze running statistics consistently and record the change.

Profile a few hundred steps before choosing the final epoch count. Estimate:

~~~text
training_seconds ≈ epochs × ceil(train_images / batch_size) × seconds_per_step
                   + validation_seconds + checkpoint_overhead
~~~

Record peak VRAM and actual GPU-hours. Do not infer feasibility from GPU model name alone.

**Outputs:** first trained checkpoint, learning curves, validation scores, and overlays.

**Exit condition:** the model trains stably, predicts ski landmarks, and has a saved reproducible baseline. An improvement over the generic model is not required on every body joint.

## 9. Step 6 — Run a small, interpretable experiment matrix

**Budget:** 20 active hours. Stop architecture exploration here unless a diagnosed failure requires a change.

Research question: **Does targeted augmentation improve ski landmarks and robustness to imperfect crops without materially degrading body accuracy?**

| ID | Model | Added training conditions | Purpose |
| --- | --- | --- | --- |
| E0 | Unchanged COCO HRNet | None | Generic body reference only |
| E1 | 24-point fine-tuning | Standard flip/rotation/scale | Main learned baseline |
| E2 | Same as E1 | Additional crop translation/scale jitter | Test tolerance to localization errors |
| E3 | Same as E1 | Motion blur and modest synthetic occlusion | Test appearance robustness |
| E4 | Same as E1 | Both E2 and E3 additions | Test the combined recipe |

E1 is the completed Step 5 run. Do not retrain it without reason.

Starting augmentation ranges:

- Rotation: approximately ±15 degrees; scale: 0.8–1.2.
- Additional crop-center jitter: up to ±10% of box width/height, with modest extra scale error.
- Motion blur: randomly oriented short kernels, approximately 3–9 pixels after model resizing.
- Occlusion: one modest patch, initially around 5–15% of crop area.
- Keep naturally uncertain labels masked. A synthetically covered point can remain supervised because its original location is known; document that policy.
- Never use horizontal mirroring without anatomical left/right label swaps.

Inspect augmented examples. Skis should remain physically interpretable and most samples should retain useful evidence. Adjust ranges using training/validation observations, then freeze them.

Report validation results on:

1. Unmodified validation images.
2. A fixed synthetic crop-error set, generated once with a recorded seed.
3. Natural difficult examples or metadata-defined strata.

Synthetic jitter measures robustness to those specific perturbations; it does not prove amateur-video generalization.

Selection rules:

- Select the final recipe on validation ski PCK, considering body degradation and crop-error robustness.
- Flag a body PCK drop above approximately two percentage points for investigation.
- The proposal’s five-percentage-point ski-PCK improvement over E1 is aspirational.
- If augmentation does not help, report it and analyze where it fails. Do not redefine the metric or test split to obtain a gain.
- If runtime is tight, preserve E1 versus E4 and at least one diagnostic ablation; explicitly disclose reduced coverage.

If compute permits, rerun E1 and the selected recipe with two additional seeds. Keep all selected seeds; do not report only the best one.

**Outputs:** experiment table, learning curves, per-landmark changes, and selected final configuration.

## 10. Step 7 — Integrate and measure tracking-to-pose error propagation

**Budget:** 16 hours.

### 7.1 Keep prediction stages inspectable

For every frame, save:

~~~text
video_id, frame_index, timestamp
tracker_box_xywh, tracker_score, tracking_status
pose_crop_transform, raw_keypoints_xy, raw_keypoint_scores
replay_crop_xywh, overlay_visibility
checkpoint_hash, configuration_hash
~~~

Keep two outputs: raw prediction data for evaluation and rendered video for interpretation. Cosmetic filtering must never overwrite raw evidence.

Smooth crop state first. Add landmark smoothing only if visible jitter remains, and evaluate it separately. If using temporal landmark filtering, retain raw accuracy and check motion lag; smoothing across turns can erase real motion.

### 7.2 Compare crops on the same labeled video frames

Use verified Ski 2DPose video segments:

1. Initialize the tracker once at the first chosen labeled frame.
2. Process **every intervening video frame**, including unlabeled frames.
3. At later labeled frames, run the same pose model with a reference crop and with a tracker-derived crop.
4. Use the same original-image labels, denominator, and evaluation-frame set.
5. Exclude initialization frames from the main propagation comparison.
6. Report `PCK_reference − PCK_tracked` and the percentage of labels excluded by the predicted crop.

Do not run the tracker only on sparse labeled images; the artificial frame gaps would change the task. Do not reset from ground truth at each labeled frame.

If only some segments align reliably, freeze that eligible subset before comparing model scores. Report the subset size and reasons for exclusion. On SkiTB, show pose overlays qualitatively unless you add pose annotations.

If alignment cannot be resolved within a short fixed budget, use synthetic crop perturbations as the quantitative diagnostic and state that true temporal propagation remains unmeasured. This is a limitation, not an equivalent replacement.

### 7.3 Diagnose failures by component

| Symptom | First question |
| --- | --- |
| Correct track, missing ski tips | Is the pose crop too tight or resolution too low? |
| Correct crop, wrong ski-side assignment | Is label semantics or occlusion the issue? |
| Good raw pose, poor overlay | Is the replay transform wrong? |
| Smooth video, skier drifts off center | Is the crop controller lagging? |
| Good reference-crop PCK, poor tracked-crop PCK | Is tracker drift or train/inference crop mismatch responsible? |
| High heatmap score on a wrong point | Is confidence being mistaken for calibrated accuracy? |

**Outputs:** integrated clips, raw predictions, a crop-propagation table, and categorized failures.

## 11. Step 8 — Freeze, evaluate, and communicate the result

**Budget:** 16 hours.

### 8.1 Freeze before final test execution

Record the split manifest, code revision, checkpoint hashes, all selected seeds, crop parameters, augmentation settings, masks, and metrics. Select two trained methods for the main test comparison: E1 and the validation-selected recipe.

Evaluate the frozen test set once for final reporting. If a genuine software bug forces a rerun, document the correction; do not silently use the test set for model selection.

### 8.2 Produce three separate quantitative tables

**Pose table**

| Method | Body PCK@0.1 | Ski PCK@0.1 | Ski PCK@0.05 | Labeled-point counts |
| --- | --- | --- | --- | --- |
| E0: shared joints only | To measure | N/A | N/A | To count |
| E1: all dataset body/ski joints | To measure | To measure | To measure | To count |
| Selected augmentation | To measure | To measure | To measure | To count |

Also include an E0/E1/selected comparison on the identical shared body-joint subset. Do not compare E0’s ten-joint number directly with E1’s fourteen-joint number as an improvement estimate.

**Tracking table:** mean IoU, per-clip IoU, failure episodes, visibility policy, frame counts, FPS.

**Replay table:** containment, centering error, acceleration, crop-area fraction, scale variation.

Add the reference-versus-tracked crop pose comparison wherever paired labels are available.

### 8.3 Express uncertainty honestly

- If multiple seeds were run, show their mean and spread.
- Show individual held-out video results because the number of source videos is small.
- For paired confidence intervals, bootstrap source-video groups, not individual neighboring frames.
- With only a few test videos, intervals can be unstable; emphasize per-video results and descriptive conclusions.
- Keep seed variability distinct from uncertainty due to the held-out sample.

### 8.4 Assemble the final package

- A clear README with setup, data-access steps, training and inference commands.
- Frozen configs, split manifests, checkpoint provenance, and metrics code.
- Raw predictions or compact result files where redistribution permits.
- Training curves, the three metric tables, and an ablation figure.
- Two or three short demonstration clips, including one failure case.
- A report explaining the problem, method, actual experiments, results, limitations, and what the evidence supports.

The project is successful if it delivers a functioning system and a technically sound learning study. A negative augmentation result can still support a strong project when measured carefully.

## 12. Step 9 — Use reserved time deliberately

**Budget:** eight reserved hours, available earlier if a blocker threatens the core.

Spend this on broken dependencies, data access, geometry bugs, or final-report work before any extension.

### First stretch: amateur follow-camera pilot

Only start after core outputs exist.

1. Choose a few short, independent clips where one skier films another.
2. Screen them for usable target size, manageable occlusion, and permitted use.
3. Run the frozen core system first.
4. If quantitative evaluation is worthwhile, label approximately 30–60 diverse frames across several clips.
5. Time ten annotations first; total effort is annotation minutes × frame count plus review.
6. Reserve whole clips/skier identities for testing if any adaptation is attempted.

With eight hours, an external evaluation pilot is usually a better target than collecting a large adaptation dataset. A 150–250-frame annotation effort may require replacing other planned work. Do not promise both within the reserved budget.

### Lower-priority stretch: 3D reconstruction

Only consider it after the amateur pilot and final core deliverables. Run a pretrained lifting or body-fitting method as an exploratory visualization. Skiing poses, occlusion, unknown cameras, and absent 3D ski ground truth make reliable validation a separate project. Do not claim measured 3D correctness from a plausible rendering.

### Lower-priority stretch: coaching feedback

Start with descriptive observations tied to visible motion, confidence, and video timestamps. Avoid presenting a 2D projected angle as a true anatomical angle or a validated technique diagnosis. A meaningful coaching evaluation would need expert judgments and an agreed rubric; without those, label feedback as preliminary.

## 13. Fallback decisions

| Risk | Trigger | Action |
| --- | --- | --- |
| SkiTB access delayed | No usable tracking data by end of Week 1 | Continue pose work; use aligned Ski 2DPose videos for demonstrations and sparse propagation evaluation. Clearly mark the benchmark gap. |
| Pose dataset unusable | No verified labels/images after Step 1 | Pause training; resolve access or choose an explicitly revised dataset/scope. Do not assume a substitute has the same ski labels. |
| MMPose installation consumes time | More than roughly 2–3 focused hours without inference | Use a known compatible course/GPU environment, or a thin PyTorch training path around official HRNet. Record the method change. |
| Tracker integration fails | More than roughly 3–4 focused hours after access | Fix the narrow adapter; if necessary use a documented available pretrained tracker and revise the baseline description. |
| GPU runtime exceeds budget | Measured run times cannot fit planned comparisons | Reduce epochs/batch footprint or experiment count uniformly; preserve the core baseline comparison. |
| Very low ski accuracy | Tiny-subset overfit passes but generalization is poor | Audit crop resolution, labels, splits, and failures before changing architectures. |
| Split coverage is inadequate | Held-out partition has almost no reliable ski labels | Rebalance by metadata before training and freeze the revised split; never optimize split choice for model scores. |
| Annotation takes too long | Pilot exceeds remaining work budget | Stop the amateur extension; keep the core complete. |

Any material change belongs in a short decision log and the final report. The proposal is the starting plan, not a reason to hide implementation constraints.

## 14. Proposed repository artifacts

These are files to create during implementation, not files already generated by this planning task.

| Path | Responsibility |
| --- | --- |
| `docs/scope.md` | Core, stretches, and exclusions |
| `docs/data_audit.md` | Counts, formats, access, alignment findings |
| `docs/evaluation_protocol.md` | Frozen metrics and masks |
| `docs/decisions.md` | Scope/configuration changes and reasons |
| `configs/` | Pose, augmentation, tracker, and renderer settings |
| `data/manifests/` | Samples, source groups, hashes, splits |
| `src/data/` | Readers and annotation conversion |
| `src/geometry/` | Crop transforms and coordinate conversion |
| `src/pose/` | Training/inference adapters |
| `src/tracking/` | Fixed tracker adapter |
| `src/video/` | Frame processing, crop control, rendering |
| `src/evaluation/` | Pose, tracking, and crop metrics |
| `scripts/` | Small commands orchestrating these modules |
| `tests/` | Geometry, masks, splits, and metric correctness |
| `results/` | Predictions, tables, figures, and logs |

Use AI assistance for bounded implementation tasks. Give it a data contract, expected inputs/outputs, and a correctness example. Inspect raw overlays and metrics yourself. The highest-risk mistakes are semantic: which joint is which, which frame a label belongs to, and whether test information leaks into decisions.

## 15. Primary sources

Checked October 17, 2026. The implementation choices, schedules, targets, and fallback policies in this document are project recommendations. Dataset facts and upstream behavior are grounded in the sources below.

- **[S1]** [EPFL Ski 2DPose dataset](https://www.epfl.ch/labs/cvlab/data/ski-2dpose-dataset/): data links, annotation scheme, visibility, and video metadata.
- **[S2]** [Official SkiTB toolkit](https://github.com/matteo-dunnhofer/SkiTB-toolkit): access, dataset layout, splits, and annotation/code license.
- **[S3]** [MMPose v1.3.2 HRNet-W32 configuration](https://github.com/open-mmlab/mmpose/blob/v1.3.2/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py): model, heatmap, preprocessing, and inherited configuration.
- **[S4]** [MMPose custom dataset guide](https://mmpose.readthedocs.io/en/latest/advanced_guides/customize_datasets.html): COCO-style adapters and metainfo.
- **[S5]** [MMPose v1.3.2 COCO-style parser](https://github.com/open-mmlab/mmpose/blob/v1.3.2/mmpose/datasets/datasets/base/base_coco_style_dataset.py): visibility handling and box parsing.
- **[S6]** [SkiTB example program](https://github.com/matteo-dunnhofer/SkiTB-toolkit/blob/main/example_skitb.py): actual CLI arguments.
- **[S7]** [SkiTB dataset reader](https://github.com/matteo-dunnhofer/SkiTB-toolkit/blob/main/toolkit/datasets/skitb.py): frame loading and camera-segment handling.
- **[S8]** [SkiTB experiment code](https://github.com/matteo-dunnhofer/SkiTB-toolkit/blob/main/toolkit/experiments/skitb.py): benchmark execution and reporting.
- **[S9]** [MMPose installation guide](https://mmpose.readthedocs.io/en/latest/installation.html) and [version FAQ](https://mmpose.readthedocs.io/en/latest/faq.html): environment compatibility.
- **[S10]** [Official HRNet implementation](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch): architectural reference and fallback implementation.

