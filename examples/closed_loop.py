"""Runnable calibration, on-policy selection, and monitored feedback control.

Run: python examples/closed_loop.py
A deterministic 2-D integrator supplies real state feedback. All policies,
costs, and states here are synthetic; no paper benchmark is being reproduced.
"""

import json

import numpy as np

from unexpired_plan import Candidate, MonitorConfig, UnexpiredPlanMonitor
from unexpired_plan.adapters import ActionNormalizer, NormalizedPolicy
from unexpired_plan.calibration import calibrate_drift_from_replay, calibrate_repetition
from unexpired_plan.selection import score_until_stable, select_candidate

HORIZON, EXECUTION_HORIZON, CALLS = 10, 2, 20
NORMALIZER = ActionNormalizer([0.0, 0.0], [0.15, 0.15])
TARGET = np.array([1.0, 0.5])


class Environment:
    """Accept normalized prefixes; convert to physical displacement once."""

    def reset(self, initial_state):
        self.position = np.array(initial_state, dtype=float, copy=True)
        self.call = 0
        return self.observation()

    def observation(self):
        return {"position": self.position.copy(), "call": self.call}

    def step(self, normalized_actions):
        self.position += NORMALIZER.denormalize(normalized_actions).sum(axis=0)
        self.call += 1
        return self.observation(), self.call >= CALLS


class Policy:
    """Plan physical displacements with a synthetic feedback gain."""

    def __init__(self, multiplier=1.0, fault_call=None):
        self.multiplier = multiplier
        self.fault_call = fault_call

    def __call__(self, observation):
        x = observation["position"].copy()
        actions = []
        for _ in range(HORIZON):
            action = self.multiplier * np.clip(0.18 * (TARGET - x), -0.15, 0.15)
            actions.append(action)
            x += action
        chunk = np.asarray(actions)
        if observation["call"] == self.fault_call:
            # A declared out-of-calibration perturbation to exercise fallback.
            chunk += np.array([0.08, -0.08])
        return chunk


def reference_replay(reference, initial_states):
    episodes = []
    for state in initial_states:
        reference.reset()
        env = Environment()
        observation = env.reset(state)
        chunks = []
        for _ in range(CALLS):
            plan = reference(observation)
            chunks.append(plan.copy())
            observation, done = env.step(plan[:EXECUTION_HORIZON])
            if done:
                break
        episodes.append(chunks)
    return episodes


def run():
    rng = np.random.default_rng(7)
    reference = NormalizedPolicy(Policy(), NORMALIZER)
    replay = reference_replay(reference, rng.uniform(-0.4, 0.0, size=(100, 2)))
    repetition = calibrate_repetition(replay)
    drift = calibrate_drift_from_replay(replay, HORIZON, EXECUTION_HORIZON)

    # Separate initial states for selection. Each candidate controls its OWN
    # environment; the reference is called there only for paid offline scoring.
    batches = [rng.uniform(-0.4, 0.0, size=(10, 2)) for _ in range(10)]
    policies = [
        Candidate("fast", NormalizedPolicy(Policy(1.03), NORMALIZER), 0.2),
        Candidate("conservative", NormalizedPolicy(Policy(1.005), NORMALIZER), 0.6),
    ]
    records = []
    for candidate in policies:
        result = score_until_stable(
            Environment,
            candidate.predict,
            reference,
            batches,
            execution_horizon=EXECUTION_HORIZON,
            max_calls=CALLS,
        )
        records.append({"name": candidate.name, "cost": candidate.cost, **result})
    selected = select_candidate(records)
    if selected is None:
        raise RuntimeError("No synthetic candidate cleared the deviation threshold")

    # Selection has finished. Inject a visible new disturbance at deployment.
    policies[0].predict.predict.fault_call = 3
    start = next(i for i, candidate in enumerate(policies) if candidate.name == selected["name"])
    monitor = UnexpiredPlanMonitor(
        reference,
        policies[start:],
        MonitorConfig(
            horizon=HORIZON,
            execution_horizon=EXECUTION_HORIZON,
            repetition_threshold=repetition["threshold"],
            drift_by_age=drift,
        ),
    )
    for policy in [reference, *(c.predict for c in policies)]:
        policy.reset()
    env = Environment()
    observation = env.reset([-0.2, -0.1])
    decisions = []
    for _ in range(CALLS):
        decision = monitor.step(observation)
        decisions.append(decision)
        observation, done = env.step(decision.executed)
        if done:
            break

    summary = {
        "kind": "synthetic_closed_loop_example",
        "selection_signal": "95th percentile of full-chunk on-policy deviation",
        "selected": selected,
        "selection_rounds": {r["name"]: r["rounds"] for r in records},
        "repetition_calibration": repetition,
        "drift_by_age": drift,
        "reference_calls": [d.call for d in decisions if d.reference_forward],
        "fallback_calls": [d.call for d in decisions if d.veto_reasons],
        "final_rung": monitor.rung,
        "final_distance_to_target": float(np.linalg.norm(env.position - TARGET)),
    }
    print(json.dumps(summary, indent=2, allow_nan=False))
    return summary


if __name__ == "__main__":
    run()
