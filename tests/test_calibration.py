import numpy as np
import pytest

from unexpired_plan.calibration import calibrate_drift_from_replay, calibrate_repetition
from unexpired_plan.metrics import deviation


def test_drift_replay_compares_matching_time_indices_and_ages():
    # A perfect analytic reference has zero replanning drift at every age.
    episode = [(1 + np.arange(2 * t, 2 * t + 10) / 100).reshape(10, 1) for t in range(11)]
    assert calibrate_drift_from_replay([episode], 10) == {1: 0, 2: 0, 3: 0, 4: 0}


def test_drift_direction_and_nondivisible_horizon():
    old = np.arange(1, 6, dtype=float).reshape(5, 1)
    fresh = np.ones((5, 1)) * 2
    drift = calibrate_drift_from_replay([[old, fresh]], 5)
    assert drift[1] == pytest.approx(deviation(fresh[:2], old[2:4]))


def test_calibration_uses_the_same_epsilon_as_the_monitor():
    zero, one = np.zeros((4, 1)), np.ones((4, 1))
    repetition = calibrate_repetition([[zero, one]], epsilon=2)
    drift = calibrate_drift_from_replay([[zero, one]], 4, epsilon=2)
    assert repetition["threshold"] == 2
    assert drift == {1: 1}


def test_calibration_accepts_episode_iterators():
    replay = ((np.ones((4, 1)) * k for k in [1, 2, 3]) for _ in range(3))
    assert calibrate_repetition(replay)["episodes"] == 3


@pytest.mark.parametrize("episode", [[], [np.ones((4, 1))], [np.full((4, 1), np.nan)]])
def test_calibration_never_silently_discards_short_or_invalid_episodes(episode):
    with pytest.raises(ValueError):
        calibrate_repetition([[np.ones((4, 1)), np.ones((4, 1))], episode])


def test_drift_replay_requires_observations_at_every_calibrated_age():
    with pytest.raises(ValueError, match="every reference age"):
        calibrate_drift_from_replay([[np.ones((10, 1))] * 2], 10)


def test_m_one_has_no_replanning_age_to_correct():
    assert calibrate_drift_from_replay([[np.ones((2, 1))] * 2], 2) == {}


def test_episode_false_veto_calibration_counts_episodes_not_calls():
    # One episode is long; it still contributes one minimum and one vote.
    episodes = [[np.ones((2, 1)), np.ones((2, 1)) * 2] for _ in range(99)]
    episodes.append([np.ones((2, 1))] * 40)
    result = calibrate_repetition(episodes, episode_false_veto_target=0.014)
    assert result["episodes"] == 100
    assert result["empirical_false_veto_rate"] == 0.01
    assert result["threshold"] == pytest.approx(1)
