"""A small runnable example of fallback without an extra reference call.

Run from an installed checkout: python examples/monitor_trace.py
The action streams are synthetic; this is not a benchmark experiment.
"""

import json
import numpy as np
from unexpired_plan import Candidate, MonitorConfig, UnexpiredPlanMonitor


def run():
    reference_calls = []

    def reference(t):
        reference_calls.append(t)
        return (1 + np.arange(2 * t, 2 * t + 10) / 100).reshape(10, 1)

    def stable(t):
        return (1 + np.arange(2 * t, 2 * t + 10) / 100).reshape(10, 1)

    def fast(t):
        return stable(t) + (0.3 if t == 3 else 0.005)

    monitor = UnexpiredPlanMonitor(
        reference,
        [
            Candidate("fast", fast, cost=0.2),
            Candidate("conservative", stable, cost=0.6),
        ],
        MonitorConfig(horizon=10, execution_horizon=2, repetition_threshold=0.0),
    )
    decisions = [monitor.step(t) for t in range(12)]
    for decision in decisions:
        row = decision.as_dict()
        row.pop("elapsed_seconds")
        print(json.dumps(row))
    assert reference_calls == [0, 5, 10]
    assert decisions[3].source == "unexpired_fallback"
    assert not decisions[3].reference_forward
    assert all(decision.rung_after == 1 for decision in decisions[3:])
    print(json.dumps({"reference_calls": reference_calls, "kind": "synthetic_example"}))


if __name__ == "__main__":
    run()
