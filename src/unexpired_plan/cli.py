"""Run a deterministic integration example, never a benchmark claim."""

import argparse
import json
import numpy as np
from .core import Candidate, MonitorConfig, UnexpiredPlanMonitor
from .metrics import effective_speedup


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calls", type=int, default=12)
    args = p.parse_args()
    if args.calls < 1:
        p.error("--calls must be positive")

    def reference(t):
        return (1 + 0.01 * np.arange(2 * t, 2 * t + 10)).reshape(10, 1)

    def fast(t):
        return reference(t) + (0.3 if t == 3 else 0.005)

    monitor = UnexpiredPlanMonitor(
        reference,
        [Candidate("fast", fast, 0.2), Candidate("conservative", reference, 0.6)],
        MonitorConfig(10),
    )
    print(
        json.dumps(
            {
                "kind": "synthetic_integration_example",
                "m": 5,
                "ideal_fixed_rung_speedup": effective_speedup(5, 5),
            }
        )
    )
    for t in range(args.calls):
        print(json.dumps(monitor.step(t).as_dict()))


if __name__ == "__main__":
    main()
