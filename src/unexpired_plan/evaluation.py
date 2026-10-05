"""Clustered paired bootstrap for later benchmark integration (§V)."""

from __future__ import annotations

import numpy as np


def paired_bootstrap(
    reference_success,
    candidate_success,
    initial_state_ids,
    resamples=10000,
    seed=0,
    margin_pp=2.0,
):
    r, c = np.asarray(reference_success, float), np.asarray(candidate_success, float)
    ids = np.asarray(initial_state_ids)
    if r.ndim != 1 or not len(r) or c.shape != r.shape or ids.shape != r.shape:
        raise ValueError("Require equally sized paired one-dimensional observations")
    if not np.isin(r, [0, 1]).all() or not np.isin(c, [0, 1]).all():
        raise ValueError("Success outcomes must be binary")
    if ids.dtype.kind not in "iuUS" and not (ids.dtype.kind == "f" and np.isfinite(ids).all()):
        raise ValueError("Cluster IDs must be finite numbers or strings, without missing IDs")
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if not np.isfinite(margin_pp) or margin_pp <= 0:
        raise ValueError("margin_pp must be finite and positive")
    _, inverse = np.unique(ids, return_inverse=True)
    sums = np.bincount(inverse, weights=(c - r) * 100)
    sizes = np.bincount(inverse)
    if len(sizes) < 2:
        raise ValueError("A clustered confidence interval requires at least two initial states")
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples)
    for b in range(resamples):
        idx = rng.integers(0, len(sums), len(sums))
        draws[b] = sums[idx].sum() / sizes[idx].sum()
    lo, hi = np.quantile(draws, [0.025, 0.975])
    point = float(np.mean(c - r) * 100)
    return {
        "delta_sr_pp": point,
        "ci95_pp": [float(lo), float(hi)],
        "equivalent": bool(lo > -margin_pp and hi < margin_pp),
        "pairs": len(r),
        "initial_state_clusters": len(sums),
        "resamples": resamples,
        "seed": seed,
    }
