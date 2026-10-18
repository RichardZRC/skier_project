import itertools
import json

import numpy as np
import pytest
import synth

from skier.splits import (
    FOLLOW_CAM,
    VideoStats,
    apply_split,
    make_pose_split,
    split_manifest,
    video_stats_from_records,
)
from skier.splits import main as split_main
from skier.data.ski2dpose import load_records

TYPES = ["Slalom", "Giant Slalom", "Super G", "Downhill"]


def synthetic_stats(n=16, seed=0):
    rng = np.random.default_rng(seed)
    stats = []
    for i in range(n):
        imgs = int(rng.integers(20, 300))
        disc = FOLLOW_CAM if i >= n - 2 else TYPES[i % len(TYPES)]
        stats.append(VideoStats(f"v{i:02d}", imgs, imgs * 5, frozenset({disc}), disc == FOLLOW_CAM))
    return stats


def brute_force(stats, forced_train=(), forced_test=(), aside=(), N=400, lo=2, hi=4, w=0.05, targets=(0.7, 0.15, 0.15)):
    """Plain reference implementation of the scoring rule: best score and the set of optimal (val, test)."""
    main = [s for s in stats if s.video_id not in aside]
    total = sum(s.n_images for s in main)
    types = set().union(*(s.disciplines for s in main)) - {FOLLOW_CAM}
    by_id = {s.video_id: s for s in main}
    cand = [s.video_id for s in main if s.video_id not in forced_train]
    results = []
    for kt in range(lo, hi + 1):
        for t in itertools.combinations(cand, kt):
            if not set(forced_test) <= set(t) or sum(by_id[v].n_ski_points for v in t) < N:
                continue
            for kv in range(lo, hi + 1):
                for v in itertools.combinations([c for c in cand if c not in t and c not in forced_test], kv):
                    if sum(by_id[x].n_ski_points for x in v) < N or len(main) - kt - kv < 1:
                        continue
                    frac = lambda p: sum(by_id[x].n_images for x in p) / total
                    miss = lambda p: len(types - set().union(*(by_id[x].disciplines for x in p)))
                    s = abs(frac(v) - targets[1]) + abs(frac(t) - targets[2]) + w * (miss(v) + miss(t))
                    results.append((s, tuple(sorted(v)), tuple(sorted(t))))
    best = min(r[0] for r in results)
    return best, {(r[1], r[2]) for r in results if r[0] <= best + 1e-12}


def test_matches_brute_force_on_7_videos():
    stats = synthetic_stats(7, seed=3)
    split = make_pose_split(stats, forced_train=["v00"], min_ski_points=200)
    best, optimal = brute_force(stats, forced_train=["v00"], N=200)
    assert split.score == pytest.approx(best, abs=1e-12)
    assert (tuple(split.partitions["val"]), tuple(split.partitions["test"])) in optimal


def test_matches_brute_force_with_forced_test_and_aside():
    stats = synthetic_stats(8, seed=5)
    kw = dict(forced_train=["v01"], forced_test=["v02"], min_ski_points=300)
    split = make_pose_split(stats, set_aside={"followcam": ["v06", "v07"]}, **kw)
    best, optimal = brute_force(stats, aside=("v06", "v07"), N=300, forced_train=["v01"], forced_test=["v02"])
    assert split.score == pytest.approx(best, abs=1e-12)
    assert (tuple(split.partitions["val"]), tuple(split.partitions["test"])) in optimal


def test_partitions_disjoint_cover_and_rules():
    stats = synthetic_stats(16)
    aside = [s.video_id for s in stats if s.followcam]
    split = make_pose_split(stats, forced_train=["v00", "v01"], forced_test=["v05"], set_aside={"followcam": aside})
    parts = split.partitions
    all_vids = [v for vids in parts.values() for v in vids]
    assert sorted(all_vids) == sorted(s.video_id for s in stats) and len(all_vids) == len(set(all_vids))
    assert {"v00", "v01"} <= set(parts["train"]) and "v05" in parts["test"]
    assert parts["followcam"] == sorted(aside)
    by_id = {s.video_id: s for s in stats}
    for p in ("val", "test"):
        assert 2 <= len(parts[p]) <= 4
        assert sum(by_id[v].n_ski_points for v in parts[p]) >= 400


def test_deterministic_and_seeded_ties():
    stats = [VideoStats(f"v{i}", 100, 1000, frozenset({"Slalom"}), False) for i in range(8)]  # all identical: many ties
    a = make_pose_split(stats, seed=1, min_ski_points=1)
    assert a.partitions == make_pose_split(stats, seed=1, min_ski_points=1).partitions
    assert a.n_ties > 1
    others = {tuple(make_pose_split(stats, seed=s, min_ski_points=1).partitions["test"]) for s in range(10)}
    assert len(others) > 1  # the seed actually breaks ties


@pytest.mark.parametrize("kwargs, match", [
    (dict(forced_train=["v00"], forced_test=["v00"]), "both"),
    (dict(forced_train=["nope"]), "unknown"),
    (dict(forced_test=["v00", "v01", "v02", "v03", "v04"]), "max_videos"),
    (dict(min_ski_points=10**9), "no feasible split"),
    (dict(set_aside={"train": ["v00"]}), "can't be named"),
])
def test_impossible_or_conflicting_raises(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_pose_split(synthetic_stats(16), **kwargs)


def test_too_few_candidates_raises():
    stats = synthetic_stats(5)
    with pytest.raises(ValueError):
        make_pose_split(stats, forced_train=["v00", "v01"], min_ski_points=1)  # 3 candidates < 2 + 2


# --- records -> stats -> manifest -> CLI -------------------------------------------------

TEMPLATE = "{video_id}/{segment_id}/{image_id}.png"


def make_dataset(tmp_path, videos=7):
    labels = synth.make_labels(videos=videos, segments=2, images=3)
    images = tmp_path / "images"
    synth.make_blank_images(images, labels, size=(64, 48), template=TEMPLATE)
    types = ["Slalom", "Giant Slalom", "Super G", "Downhill", "Slalom", "Giant Slalom", FOLLOW_CAM]
    info = {f"v{i}": {"video_info": {"type": types[i % len(types)]},
                      "splits_info": {f"s{s}": {"front_view": True} for s in range(2)}} for i in range(videos)}
    ann = labels["v1"]["s0"]["i0"]["annotation"]
    for row in ann:
        row[2] = 0  # one excluded record
    info_path = tmp_path / "info.json"
    info_path.write_text(json.dumps(info))
    return synth.write_labels(labels, tmp_path / "labels.json"), images, info_path


def test_stats_and_manifest(tmp_path, schema):
    labels, images, info = make_dataset(tmp_path)
    records = load_records(labels, images, info, path_template=TEMPLATE)
    stats = {s.video_id: s for s in video_stats_from_records(records, schema)}
    assert stats["v1"].n_images == 5 and stats["v0"].n_images == 6  # excluded record not counted
    assert stats["v6"].followcam and not stats["v0"].followcam
    split = make_pose_split(list(stats.values()), forced_train=["v0"], set_aside={"followcam": ["v6"]}, min_ski_points=1)
    m = split_manifest(split, records, schema)
    assert [e["sample_id"] for e in m["excluded"]] == ["v1/s0/i0"]
    n = sum(d["counts"]["images"] for d in m["partitions"].values())
    assert n == len(records) - 1
    assert m["partitions"]["followcam"]["counts"]["followcam_segments"] == 2
    assert m["partitions"]["followcam"]["counts"]["fraction_of_main"] is None
    assert m["content_sha256"] == split_manifest(split, records, schema)["content_sha256"]  # stable
    m["partitions"]["val"]["sample_ids"].pop()
    content = {p: {"videos": d["videos"], "sample_ids": d["sample_ids"]} for p, d in m["partitions"].items()}
    import hashlib
    assert hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest() != m["content_sha256"]

    apply_split(records, m)
    assert {r.split for r in records if r.source_video_id == "v0"} == {"train"}
    assert {r.split for r in records if r.source_video_id == "v6"} == {"followcam"}


def test_cli_writes_and_refuses_overwrite(tmp_path, capsys):
    labels, images, info = make_dataset(tmp_path)
    pilots = tmp_path / "pilots.txt"
    pilots.write_text("v0\n")
    out = tmp_path / "split.json"
    argv = ["--labels", str(labels), "--images-root", str(images), "--info", str(info), "--pilot-file", str(pilots),
            "--set-aside", "followcam=auto", "--min-ski-points", "1", "--path-template", TEMPLATE, "--out", str(out)]
    assert split_main(argv) == 0
    m = json.loads(out.read_text())
    assert m["partitions"]["followcam"]["videos"] == ["v6"]
    assert "v0" in m["partitions"]["train"]["videos"]
    assert "train" in capsys.readouterr().out
    assert split_main(argv) == 1  # frozen: refuses to overwrite
    assert "frozen" in capsys.readouterr().err
    assert split_main(argv + ["--force"]) == 0
    assert json.loads(out.read_text())["content_sha256"] == m["content_sha256"]  # same inputs, same split


def test_cli_requires_pilot_file(tmp_path, capsys):
    labels, images, info = make_dataset(tmp_path)
    code = split_main(["--labels", str(labels), "--images-root", str(images), "--info", str(info),
                       "--pilot-file", str(tmp_path / "missing.txt"), "--out", str(tmp_path / "s.json")])
    assert code == 1 and "pilot" in capsys.readouterr().err
