"""Deviation-only candidate scoring; episode success is never a selection input."""

from __future__ import annotations

from itertools import islice

import numpy as np

from .metrics import action_array, deviation


def tail_score(deviations, quantile=0.95):
    a = np.asarray(deviations, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("A nonempty vector of finite nonnegative deviations is required")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0,1]")
    return float(np.quantile(a, quantile))


def select_candidate(records, theta=0.15):
    """records: name, cost, deviations measured on that candidate's visited states."""
    if not np.isfinite(theta) or theta <= 0:
        raise ValueError("theta must be finite and positive")
    scored, names = [], set()
    for r in records:
        name = r.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Candidate names must be nonempty and unique")
        names.add(name)
        if r.get("state_distribution") != "on_policy":
            raise ValueError("Candidate must be scored on its own closed-loop state distribution")
        if r.get("converged") is False:
            raise ValueError("Candidate scoring has not converged")
        if not np.isfinite(r["cost"]) or r["cost"] <= 0:
            raise ValueError("Measured cost must be positive")
        scored.append({"name": r["name"], "cost": r["cost"], "tail": tail_score(r["deviations"])})
    eligible = sorted((r for r in scored if r["tail"] <= theta), key=lambda r: r["cost"])
    return eligible[0] if eligible else None


def score_closed_loop(
    env_factory,
    candidate,
    reference,
    initial_states,
    execution_horizon=2,
    max_calls=10000,
    epsilon=1e-8,
):
    """Paid offline selection: reference scored at states visited under candidate actions.

    env.reset(initial_state) -> obs; env.step(actions) -> (obs, done).
    Eq. (1) scores the COMPLETE chunks; only the first execution_horizon actions
    execute. This differs from the time-aligned prefix check used online.
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
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    scores = []
    shape = None
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
                if a.shape != r.shape or (shape is not None and a.shape != shape):
                    raise ValueError("Predictors must retain identical full-chunk shapes")
                shape = a.shape
                scores.append(deviation(a, r, epsilon))
                obs, done = env.step(a[:execution_horizon].copy())
                if done:
                    break
            else:
                raise RuntimeError("Episode exceeded max_calls during candidate selection")
        finally:
            if hasattr(env, "close"):
                env.close()
    if not scores:
        raise ValueError("At least one initial state is required")
    return scores


def score_until_stable(
    env_factory,
    candidate,
    reference,
    initial_state_batches,
    *,
    execution_horizon=2,
    max_calls=10000,
    epsilon=1e-8,
    atol=1e-3,
    stable_rounds=2,
    max_rounds=10,
):
    """Repeat on-policy scoring until the pooled 95th percentile stabilizes.

    Each batch supplies initial states for a new round. Convergence requires
    stable_rounds consecutive changes <= atol. These stopping controls are
    explicit implementation choices: the manuscript gives no numeric rule.
    Exhausting the batches or max_rounds returns converged=False; callers must
    not silently deploy an unconverged selection. No success labels are used.
    """
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    if (
        any(
            isinstance(n, bool) or not isinstance(n, int) or n < 1
            for n in (stable_rounds, max_rounds)
        )
        or max_rounds <= stable_rounds
    ):
        raise ValueError("Require max_rounds > stable_rounds >= 1, both integers")
    scores, history, stable = [], [], 0
    for states in islice(initial_state_batches, max_rounds):
        scores.extend(
            score_closed_loop(
                env_factory,
                candidate,
                reference,
                states,
                execution_horizon,
                max_calls,
                epsilon,
            )
        )
        tail = tail_score(scores)
        stable = stable + 1 if history and abs(tail - history[-1]) <= atol else 0
        history.append(tail)
        if stable >= stable_rounds:
            break
    if not history:
        raise ValueError("At least one initial-state batch is required")
    return {
        "state_distribution": "on_policy",
        "deviations": scores,
        "tail": history[-1],
        "tail_history": history,
        "rounds": len(history),
        "converged": stable >= stable_rounds,
    }
