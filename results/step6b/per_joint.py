"""Per-joint PL vs. E2 (D29 extra; descriptive, post hoc; no new evaluation).

The report says the pseudo-label student gains on ski tips, toes and heels but loses on the ski tails. This rebuilds
per-joint PCK@0.1 for every body and ski joint from the saved predictions of exactly the runs that
``results/step6b/<partition>/compare.json`` compared, checks each joint's count against the stored per-joint counts of
every run, and gives each event group's paired difference PL - E2 with D25 intervals (segments within videos;
follow-cam: videos). Not pre-registered: the per-joint view was looked at after the val and test results.

Usage (GPU machine, inside Ubuntu): python results/step6b/per_joint.py [--out results/step6b/per_joint]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from skier.eval.bootstrap import N_BOOT  # noqa: E402
from skier.eval.pose_metrics import gt_arrays, sample_hits  # noqa: E402
from skier.io_utils import ensure_dir, read_json, write_json  # noqa: E402
from skier.paths import data_root, repo_root  # noqa: E402
from skier.report.events import grouped, partition_groups, video_events  # noqa: E402
from skier.report.tables import align, load_eval, pooled  # noqa: E402
from skier.schema import load_schema  # noqa: E402

THR = 0.1
PARTITIONS = ("val", "test", "followcam")


def joint_arrays(run_dirs, by_id: dict, schema, joints) -> tuple[list, dict]:
    """``(records, {(joint name, THR): seed-pooled (hits[N], totals[N])})``; each joint's count is checked against
    the run's stored per-joint count."""
    seeds, order = [], None
    for d in run_dirs:
        m, z = load_eval(Path(d))
        ids = [str(i) for i in z["sample_ids"]]
        order = order or ids
        if ids != order:
            raise ValueError(f"{d}: sample order differs from the other runs")
        gt, rel, diag, _ = gt_arrays(align(by_id, ids))
        a = {}
        for j in joints:
            name = schema.names[j]
            h, n = sample_hits(z["clean"], gt, rel, diag, [j], THR)
            ref = m["clean"]["per_joint"][name]
            if (int(h.sum()), int(n.sum())) != (ref["hits"], ref["total"]):
                raise ValueError(f"{d} {name}: rebuilt {int(h.sum())}/{int(n.sum())}, stored {ref['hits']}/{ref['total']}")
            a[(name, THR)] = (h, n)
        seeds.append(a)
    return align(by_id, order), pooled(seeds)


def to_markdown(res: dict) -> str:
    L = ["# Per-joint PL vs. E2 (D29 extra; descriptive, post hoc)", "",
         f"PCK@{THR:g} (%), seeds pooled, reference crops; paired 95% intervals (D25: segments within videos; "
         "follow-cam: videos). Rebuilt from the runs named in each partition's compare.json; every joint's count matches "
         "the stored per-joint count of every run.", ""]
    for part, groups in res["partitions"].items():
        for gname, g in groups.items():
            L += [f"## {gname} ({', '.join(g['videos'])}; {g['n_samples']} samples)", "",
                  "| Joint | E2 | PL | Δ PL − E2 [95% CI] |", "| --- | --- | --- | --- |"]
            deltas = {d["group"]: d for d in g["deltas"]}
            for key, e2 in g["runs"]["E2"].items():
                pl, d = g["runs"]["PL"][key], deltas[key]
                L.append(f"| {key.split('@')[0]} | {e2['pct']:.1f} ({e2['hits']}/{e2['total']}) | "
                         f"{pl['pct']:.1f} ({pl['hits']}/{pl['total']}) | "
                         f"{d['delta']:+.1f} [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}] |")
            L.append("")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--out", type=Path, default=repo_root() / "results/step6b/per_joint")
    a = p.parse_args(argv)
    from skier.data.ski2dpose import load_records
    from skier.pose.eval_generic import select_partition

    schema = load_schema()
    joints = list(schema.groups["body"]) + list(schema.groups["ski"])
    s2d = data_root() / "ski2dpose"
    records = load_records(s2d / "labels/ski2dpose_labels.json", s2d / "images", s2d / "archives/ski2dpose_info.json",
                           schema=schema)
    manifest = read_json(repo_root() / "data/manifests/pose_split_v1.json")
    loc, train_locs = video_events(read_json(s2d / "archives/ski2dpose_info.json"), manifest)
    rows = [(schema.names[j], THR) for j in joints]
    res = {"note": "D29 extra, per joint; descriptive, post hoc; no new evaluation", "threshold": THR,
           "split_sha256": manifest["content_sha256"], "runs": {}, "partitions": {}}
    for part in PARTITIONS:
        cmp = read_json(repo_root() / f"results/step6b/{part}/compare.json")
        if cmp["split_sha256"] != manifest["content_sha256"]:
            raise ValueError(f"{part}: compare.json was made with another split")
        by_id = {r.sample_id: r for r in select_partition(records, manifest, part)[0]}
        recs, e2 = joint_arrays(cmp["e2"], by_id, schema, joints)
        recs_pl, pl = joint_arrays(cmp["pl"], by_id, schema, joints)
        if [r.sample_id for r in recs] != [r.sample_id for r in recs_pl]:
            raise ValueError(f"{part}: E2 and PL were scored on different samples")
        groups, by_video = partition_groups(part, manifest, loc, train_locs)
        res["runs"][part] = {"e2": cmp["e2"], "pl": cmp["pl"]}
        res["partitions"][part] = grouped(recs, {"E2": e2, "PL": pl}, rows, groups, a.n_boot, pairs=(("PL", "E2"),),
                                          by_video=by_video)
        # the per-joint cells must add up to the group cells compare.json reported
        sec = cmp["sections"]["Pose model, reference crops (clean)"]
        for gname, g in res["partitions"][part].items():
            for run in ("E2", "PL"):
                for grp in ("body", "ski"):
                    names = [schema.names[j] for j in schema.groups[grp]]
                    h = sum(g["runs"][run][f"{n}@{THR:g}"]["hits"] for n in names)
                    ref = sec[gname]["runs"][run][f"{grp}@{THR:g}"]["hits"]
                    if h != ref:
                        raise ValueError(f"{part} {gname} {run} {grp}: per-joint hits {h} vs compare.json {ref}")
    out = ensure_dir(a.out)
    write_json(res, out / "per_joint.json")
    (out / "per_joint.md").write_text(to_markdown(res), encoding="utf-8")
    print(to_markdown(res))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
