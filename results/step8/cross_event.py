"""Cross-event breakdown of the frozen results (post hoc; no new evaluation of test).

All 4 training videos come from one event (Almaty 2017 Universiade), and one of the 3 test videos is from that event
too. This splits the frozen D27 test predictions into "same event as training" and "other events", and puts val (all
other events) and the follow-cam set (another camera setup) next to them. A video's event is its ``video_info.location``
in the Ski 2DPose info JSON; "same event" means a location shared with a training video.

Everything is rebuilt from the saved ``predictions.npz`` files with the report-table code (skier.report.tables), the
full-set counts are checked against the stored ones, and the group counts must add up to them. Intervals follow D25:
segments resampled within videos (follow-cam: whole videos, D27). The grouping lives in ``skier.report.events``, which
the pseudo-label comparison (D29) shares.

Usage (GPU machine, inside Ubuntu): python results/step8/cross_event.py [--out results/step8/cross_event]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from skier.eval.bootstrap import N_BOOT  # noqa: E402
from skier.io_utils import ensure_dir, read_json, write_json  # noqa: E402
from skier.paths import data_root, repo_root, results_root  # noqa: E402
from skier.report.events import group_markdown, grouped, partition_groups, seed_arrays, video_events  # noqa: E402
from skier.report.tables import SEEDS, propagation_arrays  # noqa: E402
from skier.schema import load_schema  # noqa: E402

POSE_ROWS = (("body", 0.1), ("ski", 0.1), ("ski", 0.05))
PROP_ROWS = (("body", 0.1), ("ski", 0.1))


def pose_arrays(pose_dir: Path, by_id: dict, schema) -> tuple[list, dict]:
    """Seed-pooled per-sample hit arrays for E1 and E2 (reference crops), checked against the stored counts."""
    per, order = {}, None
    for r in ("E1", "E2"):
        recs, per[r] = seed_arrays([pose_dir / f"{r}-s{s}" for s in SEEDS], by_id, schema)
        ids = [x.sample_id for x in recs]
        order = order or ids
        if ids != order:
            raise ValueError(f"{r}: sample order differs")
    return recs, per


def to_markdown(res: dict) -> str:
    L = ["# Cross-event breakdown (post hoc; frozen predictions, no new evaluation)", "",
         f"Training event(s): {', '.join(res['training_locations'])}. Event = the info JSON's video location.",
         "PCK (%) seed-pooled, 95% interval (D25: segments within videos; follow-cam: videos).", ""]
    for section, title in (("pose", "Pose model, reference crops"),
                           ("propagation_siamfc", "System, SiamFC tracks (main result), test"),
                           ("propagation_stark", "System, STARK tracks (D28 extra), test")):
        if section not in res:
            continue
        L += [f"## {title}", ""]
        for gname, g in res[section].items():
            L += group_markdown(gname, g)
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--k-pose", type=float, default=1.1)
    p.add_argument("--out", type=Path, default=repo_root() / "results/step8/cross_event")
    a = p.parse_args(argv)
    from skier.data.ski2dpose import load_records
    from skier.pose.eval_generic import select_partition

    schema = load_schema()
    s2d = data_root() / "ski2dpose"
    info = read_json(s2d / "archives/ski2dpose_info.json")
    records = load_records(s2d / "labels/ski2dpose_labels.json", s2d / "images", s2d / "archives/ski2dpose_info.json",
                           schema=schema)
    manifest = read_json(repo_root() / "data/manifests/pose_split_v1.json")
    loc, train_locs = video_events(info, manifest)
    R = results_root()
    res = {"note": "post-hoc breakdown of frozen predictions (D27 test, val, follow-cam); no new evaluation",
           "split_sha256": manifest["content_sha256"], "training_locations": sorted(train_locs),
           "video_location": {v: loc[v] for v in sorted(loc)}, "pose": {}}
    by_part = {part: {r.sample_id: r for r in select_partition(records, manifest, part)[0]}
               for part in ("test", "val", "followcam")}
    e21 = (("E2", "E1"),)
    for part in ("test", "val", "followcam"):
        groups, by_video = partition_groups(part, manifest, loc, train_locs)
        recs, arr = pose_arrays(R / f"final/{part}/pose", by_part[part], schema)
        res["pose"].update(grouped(recs, arr, POSE_ROWS, groups, a.n_boot, e21, by_video=by_video))
    test_groups, _ = partition_groups("test", manifest, loc, train_locs)
    for name, root in (("propagation_siamfc", R / "final/test/propagation"),
                       ("propagation_stark", R / "final_stark/test/propagation")):
        if not (root / "propagation.json").is_file():
            continue
        pa = propagation_arrays(root, by_part["test"], schema, a.k_pose, ("E2",))
        arrays = {f"E2 {crop}": pa["arrays"][("E2", crop)] for crop in pa["crops"]}
        pairs = [("E2 reference", "E2 tracked")] + ([("E2 refined", "E2 tracked")] if "refined" in pa["crops"] else [])
        res[name] = grouped(pa["recs"], arrays, PROP_ROWS, test_groups, a.n_boot, pairs)
    out = ensure_dir(a.out)
    write_json(res, out / "cross_event.json")
    (out / "cross_event.md").write_text(to_markdown(res), encoding="utf-8")
    print(to_markdown(res))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
