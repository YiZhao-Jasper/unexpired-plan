"""Deviation-only candidate scoring; episode success is never a selection input."""

from __future__ import annotations
import numpy as np
from .metrics import action_array, deviation


def tail_score(deviations, quantile=0.95):
    a = np.asarray(deviations, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError(
            "A nonempty vector of finite nonnegative deviations is required"
        )
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0,1]")
    return float(np.quantile(a, quantile))


def select_candidate(records, theta=0.15):
    """records: name, cost, deviations measured on that candidate's visited states."""
    if not np.isfinite(theta) or theta <= 0:
        raise ValueError("theta must be finite and positive")
    scored = []
    for r in records:
        if r.get("state_distribution") != "on_policy":
            raise ValueError(
                "Candidate must be scored on its own closed-loop state distribution"
            )
        if not np.isfinite(r["cost"]) or r["cost"] <= 0:
            raise ValueError("Measured cost must be positive")
        scored.append(
            {"name": r["name"], "cost": r["cost"], "tail": tail_score(r["deviations"])}
        )
    eligible = sorted(
        (r for r in scored if r["tail"] <= theta), key=lambda r: r["cost"]
    )
    return eligible[0] if eligible else None


def score_closed_loop(
    env_factory,
    candidate,
    reference,
    initial_states,
    execution_horizon=2,
    max_calls=10000,
):
    """Paid offline selection: reference scored at states visited under candidate actions.

    env.reset(initial_state) -> obs; env.step(actions) -> (obs, done).
    Predictors with reset() are reset for each episode. Other stateful predictors
    must be reset by the caller's adapter. No success labels are read. The bound
    max_calls prevents a broken environment from running indefinitely.
    """
    if (
        isinstance(execution_horizon, bool)
        or not isinstance(execution_horizon, int)
        or execution_horizon < 1
    ):
        raise ValueError("execution_horizon must be a positive integer")
    if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1:
        raise ValueError("max_calls must be a positive integer")
    scores = []
    for state in initial_states:
        for p in (candidate, reference):
            if hasattr(p, "reset"):
                p.reset()
        env = env_factory()
        try:
            obs = env.reset(state)
            for _ in range(max_calls):
                a = action_array(candidate(obs)).copy()
                r = action_array(reference(obs))
                if min(len(a), len(r)) < execution_horizon:
                    raise ValueError("Both predictors must cover the execution horizon")
                scores.append(deviation(a[:execution_horizon], r[:execution_horizon]))
                obs, done = env.step(a[:execution_horizon].copy())
                if done:
                    break
            else:
                raise RuntimeError(
                    "Episode exceeded max_calls during candidate selection"
                )
        finally:
            if hasattr(env, "close"):
                env.close()
    return scores
