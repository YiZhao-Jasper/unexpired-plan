"""Deterministic synthetic trace; no pretrained model or benchmark results."""

import numpy as np

from .core import Candidate, MonitorConfig, UnexpiredPlanMonitor


def run_demo(calls=12):
    """Return decisions and actual reference invocation indices for inspection."""
    if isinstance(calls, bool) or not isinstance(calls, int) or calls < 1:
        raise ValueError("calls must be a positive integer")
    reference_calls = []

    def plan(t):
        # An analytic stream, not a model forward pass hidden in a candidate.
        return (1 + 0.01 * np.arange(2 * t, 2 * t + 10)).reshape(10, 1)

    def reference(t):
        reference_calls.append(t)
        return plan(t)

    def fast(t):
        return plan(t) + (0.3 if t == 3 else 0.005)

    monitor = UnexpiredPlanMonitor(
        reference,
        [Candidate("fast", fast, 0.2), Candidate("conservative", plan, 0.6)],
        # This trace isolates deviation fallback. The closed-loop example
        # calibrates and enables both checks from separate reference replay.
        MonitorConfig(horizon=10, repetition_threshold=0.0),
    )
    return [monitor.step(t) for t in range(calls)], reference_calls
