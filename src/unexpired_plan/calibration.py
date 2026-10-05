"""Reference replay calibration; the PDF does not give theta_lambda numerically."""

from __future__ import annotations

import numpy as np

from .metrics import _horizons, action_array, deviation


def _replay_episodes(reference_episodes, horizon=None):
    """Validate every chunk, including episodes too short for a comparison."""
    shape = None
    for episode in reference_episodes:
        chunks = []
        for value in episode:
            chunk = action_array(value).copy()
            if (horizon is not None and len(chunk) != horizon) or (
                shape is not None and chunk.shape != shape
            ):
                raise ValueError("Replay chunks must have the same full-chunk shape")
            shape = chunk.shape
            chunks.append(chunk)
        if len(chunks) < 2:
            raise ValueError("Every replay episode requires at least two chunks")
        yield chunks


def calibrate_repetition(
    reference_episodes, window=8, episode_false_veto_target=0.014, epsilon=1e-8
):
    """Lower tail of episode minima, strictly '<' veto.

    This explicit implementation choice controls EMPIRICAL reference replay
    false vetoes. It does not claim distribution-free guarantees.
    """
    if (
        isinstance(window, bool)
        or not isinstance(window, int)
        or window < 1
        or not 0 <= episode_false_veto_target < 1
        or not np.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError("Invalid calibration controls")
    minima = []
    for episode in _replay_episodes(reference_episodes):
        minimum = float("inf")
        for t, a in enumerate(episode):
            for h in episode[max(0, t - window) : t]:
                minimum = min(minimum, deviation(a, h, epsilon))
        minima.append(minimum)
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


def calibrate_drift_from_replay(
    reference_episodes, horizon, execution_horizon=2, *, quantile=0.5, epsilon=1e-8
):
    """Compare replanned prefixes with aligned saved reference slices by age.

    Replay must contain a fresh reference prediction at EVERY control call,
    collected while executing reference actions. Anchors are calls 0, m, 2m, ...;
    row offsets are age * execution_horizon, exactly as in the online monitor.
    The median estimator is an explicit choice, not an unpublished paper value.
    """
    m = _horizons(horizon, execution_horizon)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not 0 <= quantile <= 1:
        raise ValueError("Invalid quantile")
    samples = {age: [] for age in range(1, m)}
    episodes = 0
    for chunks in _replay_episodes(reference_episodes, horizon):
        episodes += 1
        for start in range(0, len(chunks), m):
            for age in range(1, min(m, len(chunks) - start)):
                offset = age * execution_horizon
                samples[age].append(
                    deviation(
                        chunks[start + age][:execution_horizon],
                        chunks[start][offset : offset + execution_horizon],
                        epsilon,
                    )
                )
    if not episodes:
        raise ValueError("At least one replay episode is required")
    if any(not values for values in samples.values()):
        raise ValueError("Replay must cover every reference age from 1 to m-1")
    return calibrate_drift(samples, quantile)
