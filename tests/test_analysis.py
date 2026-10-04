import json
from pathlib import Path
import numpy as np
import pytest
from unexpired_plan.metrics import (
    deviation,
    effective_speedup,
    coverage,
    expected_damage,
    exposure_step_ratio,
    composite_miss_rate,
)
from unexpired_plan.selection import tail_score, select_candidate, score_closed_loop
from unexpired_plan.calibration import calibrate_repetition
from unexpired_plan.evaluation import paired_bootstrap


def test_deviation_uses_reference_denominator():
    assert deviation([[3, 1]], [[1, 1]]) == pytest.approx(1)
    with pytest.raises(ValueError):
        deviation([[1]], [[1, 2]])


def test_ceiling_and_coverage_are_counting_identities():
    assert effective_speedup(5, float("inf")) == 5
    assert effective_speedup(2, 1) == 1
    assert coverage(10, 2) == {
        "m": 5,
        "period": 5,
        "free_all_calls": 0.8,
        "guarded_all_calls": 1,
        "checked_accelerated_calls": 1,
    }
    assert coverage(10, 2, 6)["checked_accelerated_calls"] == 0.8
    assert coverage(10, 2, 6)["guarded_all_calls"] == pytest.approx(5 / 6)


def test_exposure_ratio_matches_equation_four():
    e0 = expected_damage(5, 5, 200, 0.04, 0.045)
    e1 = expected_damage(5, 6, 200, 0.04, 0.045)
    assert e1 / e0 == pytest.approx(exposure_step_ratio(5, 0.045))
    assert expected_damage(5, 5, 200, 0, 0.045) == 0


def test_composite_rate_and_dependence_bounds_match_paper_rounding():
    r = composite_miss_rate([0.25, 0.12, 0.63], [1, 0.50, 0.02], [0.06, 0.29, 1])
    assert r["independent"] == pytest.approx(0.045)
    assert r["lower"] == pytest.approx(0.0276)
    assert r["upper"] == pytest.approx(0.0624)


def test_tail_scoring_catches_sparse_bad_deviations():
    assert tail_score([0.01] * 93 + [0.5] * 7) > 0.15
    assert np.mean([0.01] * 93 + [0.5] * 7) < 0.15


def test_selection_requires_own_state_distribution_and_ignores_success():
    rows = [
        {
            "name": "fast_bad",
            "cost": 1,
            "deviations": [0.5] * 20,
            "state_distribution": "on_policy",
            "success": 1,
        },
        {
            "name": "slow_good",
            "cost": 2,
            "deviations": [0.01] * 20,
            "state_distribution": "on_policy",
            "success": 0,
        },
    ]
    assert select_candidate(rows)["name"] == "slow_good"
    rows[1]["state_distribution"] = "reference_replay"
    with pytest.raises(ValueError):
        select_candidate(rows)


def test_closed_loop_selection_reference_sees_candidate_visited_states():
    seen = []

    class Env:
        def reset(self, state):
            self.x = state
            return self.x

        def step(self, a):
            self.x += float(a.sum())
            return self.x, self.x >= 4

    def ref(x):
        seen.append(x)
        return np.ones((2, 1))

    score_closed_loop(Env, lambda x: np.ones((2, 1)) * 2, ref, [0], 2)
    assert seen == [0]


def test_reference_calibration_empirical_rate_and_strict_threshold():
    episodes = [[np.ones((2, 1)), np.ones((2, 1)) * (1 + i / 100)] for i in range(100)]
    result = calibrate_repetition(episodes, episode_false_veto_target=0.014)
    assert result["episodes"] == 100
    assert result["empirical_false_veto_rate"] <= 0.014


def test_paired_cluster_bootstrap_identity():
    r = paired_bootstrap([1, 0, 1, 1], [1, 0, 1, 1], ["a", "a", "b", "b"], 100, seed=7)
    assert r["delta_sr_pp"] == 0 and r["ci95_pp"] == [0, 0] and r["equivalent"]
    assert r["initial_state_clusters"] == 2


def test_paper_results_remain_labeled_as_reported():
    root = Path(__file__).resolve().parents[1]
    claims = json.loads((root / "research/reported_results.json").read_text())
    assert "not reproduced" in claims["provenance"]
    assert len(claims["families"]) == 5
    assert claims["families"][2]["within_margin"] is False
    assert claims["families"][4]["paired_episodes"] == 6000


@pytest.mark.parametrize("margin", [0, -1, float("nan"), float("inf")])
def test_bootstrap_rejects_invalid_equivalence_margin(margin):
    with pytest.raises(ValueError):
        paired_bootstrap([1, 0], [1, 0], ["a", "b"], resamples=10, margin_pp=margin)
