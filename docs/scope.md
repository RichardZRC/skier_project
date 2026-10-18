# Scope

Copied from the [six-week plan](../skier_implementation_plan.md) §1 (v2, 2026-10-17). Changes to scope are logged in [decisions.md](decisions.md).

**Core deliverable:** given a single-camera skiing video and a box around the skier in the first usable frame, produce a smooth centered replay with 2D body and ski landmarks, plus reproducible quantitative results.

**Learning contribution (option A, chosen 2026-10-17):** fine-tune a COCO-pretrained HRNet-W32 to predict 24 body/ski/pole landmarks, and run a controlled study of whether targeted augmentation (crop jitter, motion blur, occlusion) improves ski-landmark accuracy and robustness to imperfect crops. The tracker is a fixed, pretrained SiamFC. The replay is geometry, not a learned model.

**Research question:** does targeted augmentation improve ski landmarks and robustness to imperfect crops, without materially hurting body accuracy?

Priority order:

1. **Core:** tracking, 2D body/ski landmarks, centered replay, evaluation, report.
2. **First stretch:** amateur footage filmed by another moving skier.
3. **Lower-priority stretches:** exploratory 3D, preliminary coaching feedback.

**Out of scope for the core:** web app, automatic skier selection, multi-person tracking, recovery across broadcast camera cuts, real-time speed, training a tracker.

Pose runs on the **original frame** through its own crop. The replay crop is applied afterwards, so changing replay smoothing never changes pose predictions. The replay only repositions pixels; it cannot undo blur or reveal occluded skis.
