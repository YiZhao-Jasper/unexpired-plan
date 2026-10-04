"""Reference replay calibration; the PDF does not give theta_lambda numerically."""

from __future__ import annotations
import numpy as np
from .metrics import deviation


def calibrate_repetition(reference_episodes, window=8, episode_false_veto_target=0.014):
    """Lower tail of episode minima, strictly '<' veto.

    This explicit implementation choice controls EMPIRICAL reference replay
    false vetoes. It does not claim distribution-free guarantees.
    """
    if (
        isinstance(window, bool)
        or not isinstance(window, int)
        or window < 1
        or not 0 <= episode_false_veto_target < 1
    ):
        raise ValueError("Invalid calibration controls")
    minima = []
    for episode in reference_episodes:
        vals = []
        for t, a in enumerate(episode):
            vals.extend(deviation(a, h) for h in episode[max(0, t - window) : t])
        if vals:
            minima.append(min(vals))
    if not minima:
        raise ValueError("Reference replay requires episodes with at least two chunks")
    ordered = np.sort(minima)
    # Largest order statistic below which at most floor(alpha*N) episodes lie.
    rank = int(np.floor(episode_false_veto_target * len(ordered)))
    threshold = float(ordered[rank])
    return {
        "threshold": threshold,
        "episodes": len(ordered),
        "empirical_false_veto_rate": float(np.mean(ordered < threshold)),
        "target": episode_false_veto_target,
    }


def calibrate_drift(samples_by_age, quantile=0.5):
    """Median normalized replay drift by age; quantile is an explicit extension."""
    if not 0 <= quantile <= 1:
        raise ValueError("Invalid quantile")
    out = {}
    for age, samples in samples_by_age.items():
        a = np.asarray(samples, dtype=float)
        if (
            isinstance(age, bool)
            or not isinstance(age, int)
            or age < 1
            or a.ndim != 1
            or not a.size
            or not np.isfinite(a).all()
            or (a < 0).any()
        ):
            raise ValueError("Invalid reference drift sample")
        out[int(age)] = float(np.quantile(a, quantile))
    return out
