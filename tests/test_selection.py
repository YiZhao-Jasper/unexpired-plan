import numpy as np
import pytest

from unexpired_plan.selection import score_closed_loop, score_until_stable, select_candidate


class OneCall:
    def reset(self, state):
        return state

    def step(self, actions):
        assert actions.shape == (2, 1)
        return 0, True


def test_selection_scores_full_plan_even_when_executed_prefix_matches():
    candidate = lambda _: np.array([[1], [1], [2], [2]])
    reference = lambda _: np.ones((4, 1))
    scores = score_closed_loop(OneCall, candidate, reference, [0])
    assert scores == pytest.approx([0.5])
    assert (
        select_candidate(
            [
                {
                    "name": "bad_tail",
                    "cost": 1,
                    "deviations": scores,
                    "state_distribution": "on_policy",
                }
            ]
        )
        is None
    )


def test_candidate_controls_every_visited_state_and_lifecycle_is_reset():
    reference_seen, closed, resets = [], [], []

    class Environment:
        def reset(self, state):
            self.x = state
            self.calls = 0
            return self.x

        def step(self, actions):
            self.x += float(actions.sum())
            self.calls += 1
            return self.x, self.calls == 3

        def close(self):
            closed.append(True)

    class Predictor:
        def __init__(self, is_reference):
            self.is_reference = is_reference

        def reset(self):
            resets.append(self.is_reference)

        def __call__(self, x):
            if self.is_reference:
                reference_seen.append(x)
            return np.ones((4, 1)) * (1 if self.is_reference else 2)

    score_closed_loop(Environment, Predictor(False), Predictor(True), [0, 10])
    assert reference_seen == [0, 4, 8, 10, 14, 18]
    assert resets == [False, True, False, True]
    assert closed == [True, True]


def test_offline_scoring_snapshots_shared_predictor_buffer():
    buffer = np.zeros((4, 1))

    def candidate(_):
        buffer[:] = 2
        return buffer

    def reference(_):
        buffer[:] = 1
        return buffer

    assert score_closed_loop(OneCall, candidate, reference, [0]) == pytest.approx([1])


def test_shape_mismatch_is_rejected_before_environment_execution():
    class NoExecution(OneCall):
        def step(self, actions):
            raise AssertionError("Mismatched plans must not execute")

    with pytest.raises(ValueError, match="full-chunk shapes"):
        score_closed_loop(NoExecution, lambda _: np.ones((5, 1)), lambda _: np.ones((4, 1)), [0])


def test_selection_requires_initial_states():
    with pytest.raises(ValueError, match="initial state"):
        score_closed_loop(OneCall, lambda _: np.ones((4, 1)), lambda _: np.ones((4, 1)), [])


def test_tail_stability_repeats_scoring_and_terminates_without_success_labels():
    result = score_until_stable(
        OneCall,
        lambda _: np.ones((4, 1)) * 1.01,
        lambda _: np.ones((4, 1)),
        [[0], [1], [2], [3]],
        stable_rounds=2,
    )
    assert result["converged"] and result["rounds"] == 3
    assert len(result["deviations"]) == 3
    assert result["tail"] == pytest.approx(0.01)


def test_exhausted_batches_do_not_silently_claim_convergence():
    result = score_until_stable(
        OneCall,
        lambda _: np.ones((4, 1)),
        lambda _: np.ones((4, 1)),
        [[0]],
    )
    assert not result["converged"]
    with pytest.raises(ValueError, match="not converged"):
        select_candidate([{"name": "test", "cost": 1, **result}])


def test_scoring_round_limit_bounds_an_infinite_batch_source():
    from itertools import repeat

    result = score_until_stable(
        OneCall,
        lambda x: np.ones((4, 1)) * (1 + x),
        lambda _: np.ones((4, 1)),
        ([x] for x, _ in enumerate(repeat(None))),
        atol=0,
        max_rounds=4,
    )
    assert result["rounds"] == 4 and not result["converged"]


def test_custom_epsilon_is_used_in_offline_scoring():
    scores = score_closed_loop(
        OneCall,
        lambda _: np.ones((4, 1)),
        lambda _: np.zeros((4, 1)),
        [0],
        epsilon=2,
    )
    assert scores == [2]


@pytest.mark.parametrize("names", [["same", "same"], ["", "valid"]])
def test_ambiguous_candidate_names_cannot_select_a_ladder(names):
    with pytest.raises(ValueError, match="names"):
        select_candidate(
            [
                {"name": name, "cost": 1, "deviations": [0.01], "state_distribution": "on_policy"}
                for name in names
            ]
        )
