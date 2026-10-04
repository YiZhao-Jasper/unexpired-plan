"""Exercise the calibration and selection APIs on synthetic replay data."""

import json
import numpy as np
from unexpired_plan.calibration import calibrate_repetition
from unexpired_plan.selection import select_candidate


def run():
    # Replace with full normalized chunks from unaccelerated reference episodes.
    replay = [
        [np.ones((10, 2)), np.ones((10, 2)) * (1 + i / 1000)] for i in range(1, 101)
    ]
    calibration = calibrate_repetition(replay, episode_false_veto_target=0.014)
    # Replace with measured costs and deviations on each candidate's own states.
    records = [
        {
            "name": "fast",
            "cost": 0.2,
            "deviations": [0.02] * 93 + [0.5] * 7,
            "state_distribution": "on_policy",
        },
        {
            "name": "conservative",
            "cost": 0.6,
            "deviations": [0.04] * 100,
            "state_distribution": "on_policy",
        },
    ]
    print(
        json.dumps(
            {
                "kind": "synthetic_example",
                "calibration": calibration,
                "selected": select_candidate(records),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
