"""Cluster bootstrap intervals (plan Step 8; D25).

Items (labels, frames, clips) belong to clusters (Ski 2DPose segments, SkiTB sequences).
A resample draws clusters with replacement, within each stratum (the source video) when
strata are given, and every item of a drawn cluster enters once per draw. The statistic
is recomputed from pooled counts in each resample. Intervals are percentile intervals.

D25 settings: 10,000 resamples, seed 230, 95%. With strata the videos are held fixed, so
an interval is conditional on the videos in the partition.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

N_BOOT = 10_000
SEED = 230
LEVEL = 0.95


def _cluster_index(clusters: Sequence, strata: Sequence | None):
    """``{stratum: [item-index array per cluster]}``; refuses a cluster split across strata."""
    clusters = np.asarray(clusters)
    strata_arr = np.zeros(len(clusters), dtype=int) if strata is None else np.asarray(strata)
    if len(strata_arr) != len(clusters):
        raise ValueError(f"{len(clusters)} cluster labels but {len(strata_arr)} strata labels")
    out: dict = {}
    for c in dict.fromkeys(clusters.tolist()):  # first-seen order, so a seed always gives the same draws
        idx = np.flatnonzero(clusters == c)
        s = set(strata_arr[idx].tolist())
        if len(s) != 1:
            raise ValueError(f"cluster {c!r} spans several strata: {sorted(map(str, s))}")
        out.setdefault(s.pop(), []).append(idx)
    for s, members in out.items():
        if len(members) < 2:
            raise ValueError(f"stratum {s!r} has {len(members)} cluster(s); a bootstrap needs at least 2")
    return out


def cluster_bootstrap(stat_fn: Callable[[np.ndarray], float], clusters: Sequence, strata: Sequence | None = None,
                      n_boot: int = N_BOOT, seed: int = SEED, level: float = LEVEL) -> dict:
    """``stat_fn(idx)``: the statistic over the items ``idx`` (an index array, repeats allowed)."""
    groups = _cluster_index(clusters, strata)
    point = float(stat_fn(np.arange(len(clusters))))
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    for b in range(n_boot):
        parts = []
        for members in groups.values():
            for k in rng.integers(0, len(members), size=len(members)):
                parts.append(members[k])
        values[b] = stat_fn(np.concatenate(parts))
    finite = values[np.isfinite(values)]
    lo, hi = np.percentile(finite, [100 * (1 - level) / 2, 100 * (1 + level) / 2]) if len(finite) else (np.nan, np.nan)
    return {"estimate": point, "ci": [float(lo), float(hi)], "level": level, "n_boot": n_boot, "seed": seed,
            "n_clusters": int(sum(len(m) for m in groups.values())), "n_strata": len(groups),
            "n_nonfinite": int(n_boot - len(finite))}


def ratio_ci(num, den, clusters, strata=None, scale: float = 100.0, **kw) -> dict:
    """Interval for ``scale · Σnum / Σden`` (a pooled rate such as PCK or mean IoU)."""
    num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    return cluster_bootstrap(lambda i: scale * num[i].sum() / den[i].sum(), clusters, strata, **kw)


def ratio_diff_ci(num_a, den_a, num_b, den_b, clusters, strata=None, scale: float = 100.0, **kw) -> dict:
    """Paired interval for ``scale · (Σnum_a / Σden_a − Σnum_b / Σden_b)`` over the same items."""
    na, da, nb, db = (np.asarray(x, dtype=np.float64) for x in (num_a, den_a, num_b, den_b))
    return cluster_bootstrap(lambda i: scale * (na[i].sum() / da[i].sum() - nb[i].sum() / db[i].sum()),
                             clusters, strata, **kw)
