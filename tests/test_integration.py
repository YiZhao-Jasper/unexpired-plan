"""Check failure handling and lifecycle at public API boundaries."""

import numpy as np
import pytest

from unexpired_plan import Candidate, MonitorConfig, UnexpiredPlanMonitor
from unexpired_plan.calibration import calibrate_drift, calibrate_repetition
from unexpired_plan.metrics import deviation
from unexpired_plan.selection import score_closed_loop


def test_failed_prediction_preserves_reference_and_history():
    reference = lambda _: np.ones((6, 1))
    monitor = UnexpiredPlanMonitor(
        reference,
        [
            Candidate("drifts", lambda _: np.ones((6, 1)) * 2, 0.2),
            Candidate("invalid", lambda _: np.full((6, 1), np.nan), 0.6),
        ],
        MonitorConfig(6, repetition_window=1),
    )
    monitor.step(0)
    monitor.step(1)
    history = [(t, a.copy()) for t, a in monitor.history]
    with pytest.raises(ValueError):
        monitor.step(2)
    assert monitor.call == 2 and monitor.rung == 1
    assert [t for t, _ in monitor.history] == [t for t, _ in history]
    for (_, actual), (_, expected) in zip(monitor.history, history):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(monitor.plan, np.ones((6, 1)))


def test_calibration_config_is_not_changed_by_mutating_input():
    drift = {1: 0.05}
    config = MonitorConfig(6, drift_by_age=drift)
    drift[1] = 100
    assert config.drift_by_age[1] == 0.05
    with pytest.raises(TypeError):
        config.drift_by_age[1] = 100


def test_boolean_horizon_is_not_an_action_count():
    with pytest.raises(ValueError):
        MonitorConfig(True, execution_horizon=1)


def test_large_finite_inputs_do_not_silently_produce_nan():
    assert deviation([[1e308, 1e308]], [[1e308, 1e308]]) == 0
    assert deviation([[1e308]], [[-1e308]]) == pytest.approx(2)
    with pytest.raises(ValueError):
        deviation([[1e308]], [[0]])


def test_full_chunks_expire_by_control_call_even_after_fallbacks():
    candidates = [
        Candidate(str(i), lambda _: np.ones((20, 1)) * 2, i + 1) for i in range(10)
    ]
    monitor = UnexpiredPlanMonitor(
        lambda _: np.ones((20, 1)), candidates, MonitorConfig(20)
    )
    for t in range(9):
        monitor.step(t)
    assert [t for t, _ in monitor.history] == [0]
    monitor.step(9)
    assert len(monitor.history) == 0


def test_tail_scoring_terminates_broken_environment_and_closes_it():
    closed = []

    class Environment:
        def reset(self, state):
            return state

        def step(self, actions):
            return 0, False

        def close(self):
            closed.append(True)

    predictor = lambda _: np.ones((2, 1))
    with pytest.raises(RuntimeError, match="max_calls"):
        score_closed_loop(Environment, predictor, predictor, [0], max_calls=3)
    assert closed == [True]


def test_tail_scoring_rejects_truncated_action_prefix():
    class Environment:
        def reset(self, state):
            return state

        def step(self, actions):
            raise AssertionError("Invalid actions must not execute")

    with pytest.raises(ValueError, match="execution horizon"):
        score_closed_loop(
            Environment, lambda _: np.ones((1, 1)), lambda _: np.ones((2, 1)), [0]
        )


@pytest.mark.parametrize("window", [0, 1.5, True])
def test_repetition_calibration_requires_integer_window(window):
    with pytest.raises(ValueError):
        calibrate_repetition([[np.ones((2, 1)), np.ones((2, 1))]], window=window)


def test_drift_calibration_does_not_silently_truncate_fractional_age():
    with pytest.raises(ValueError):
        calibrate_drift({1.5: [0.01, 0.02]})
