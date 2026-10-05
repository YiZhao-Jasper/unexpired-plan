import numpy as np
import pytest

from unexpired_plan import Candidate, MonitorConfig, UnexpiredPlanMonitor


class Reference:
    def __init__(self, h=10):
        self.calls = []
        self.h = h

    def __call__(self, t):
        self.calls.append(t)
        return (1 + np.arange(t * 2, t * 2 + self.h) / 100).reshape(self.h, 1)


def predictor(t, h=10):
    return (1 + np.arange(t * 2, t * 2 + h) / 100).reshape(h, 1)


def test_reference_is_paid_once_per_m_and_exact_indices_are_compared():
    ref = Reference()
    mon = UnexpiredPlanMonitor(ref, [Candidate("exact", predictor, 0.3)], MonitorConfig(10))
    out = [mon.step(t) for t in range(15)]
    assert ref.calls == [0, 5, 10]
    for t, decision in enumerate(out):
        np.testing.assert_allclose(decision.executed, predictor(t)[:2])
        assert decision.delta_raw in (None, 0.0)


def test_veto_executes_saved_slice_without_calling_reference():
    ref = Reference()
    mon = UnexpiredPlanMonitor(
        ref,
        [
            Candidate("bad", lambda t: predictor(t) + 1, 0.2),
            Candidate("slow", predictor, 0.7),
        ],
        MonitorConfig(10),
    )
    mon.step(0)
    out = mon.step(1)
    assert ref.calls == [0]
    assert out.source == "unexpired_fallback"
    assert out.rung_before == 0 and out.rung_after == 1
    np.testing.assert_allclose(out.executed, predictor(1)[:2])
    assert out.veto_reasons == ("plan_deviation",)


def test_demotion_persists_across_reference_refresh_and_resets_only_at_episode():
    mon = UnexpiredPlanMonitor(
        Reference(),
        [
            Candidate("bad", lambda t: predictor(t) + 1, 0.2),
            Candidate("slow", predictor, 0.7),
        ],
        MonitorConfig(10),
    )
    out = [mon.step(t) for t in range(11)]
    assert all(d.rung_after == 1 for d in out[1:])
    mon.reset()
    assert mon.rung == 0 and mon.call == 0 and len(mon.history) == 0


def test_ladder_exhaustion_returns_to_reference_every_call():
    ref = Reference()
    mon = UnexpiredPlanMonitor(
        ref, [Candidate("bad", lambda t: predictor(t) + 1, 0.1)], MonitorConfig(10)
    )
    out = [mon.step(t) for t in range(5)]
    assert ref.calls == [0, 2, 3, 4]
    assert out[1].source == "unexpired_fallback"
    assert all(d.reference_forward for d in out[2:])


def test_repetition_detects_the_plan_checks_blind_spot():
    # A constant old plan has delta=0 against itself and lambda=0 against history.
    plan = np.ones((10, 2))
    mon = UnexpiredPlanMonitor(
        lambda _: plan,
        [Candidate("reuse", lambda _: plan, 0.1, "reuse")],
        MonitorConfig(10, repetition_threshold=0.001),
    )
    mon.step(None)
    decision = mon.step(None)
    assert decision.delta_raw == 0 and decision.repetition == 0
    assert decision.veto_reasons == ("chunk_repetition",)
    assert decision.source == "unexpired_fallback"


def test_recompute_only_drift_correction():
    def run(mechanism):
        mon = UnexpiredPlanMonitor(
            lambda _: np.ones((4, 1)),
            [Candidate(mechanism, lambda _: np.ones((4, 1)) * 1.2, 0.3, mechanism)],
            MonitorConfig(4, drift_by_age={1: 0.1}),
        )
        mon.step(0)
        return mon.step(1)

    assert run("recompute").source == "accelerated"
    assert run("reuse").source == "unexpired_fallback"
    assert run("blend").source == "unexpired_fallback"


def test_nondivisible_horizon_never_reads_an_expired_slice():
    ref = Reference(5)
    mon = UnexpiredPlanMonitor(
        ref, [Candidate("exact", lambda t: predictor(t, 5), 0.2)], MonitorConfig(5)
    )
    out = [mon.step(t) for t in range(6)]
    assert ref.calls == [0, 2, 4]
    assert all(d.executed.shape == (2, 1) for d in out)


def test_m_one_always_runs_reference():
    ref = Reference(2)

    def forbidden(t):
        raise AssertionError("No acceleration slot when m=1")

    mon = UnexpiredPlanMonitor(ref, [Candidate("unused", forbidden, 0.1)], MonitorConfig(2))
    assert all(mon.step(t).reference_forward for t in range(5))


@pytest.mark.parametrize("value", [np.full((10, 1), np.nan), np.ones((3, 1)), np.ones((10, 2))])
def test_invalid_candidate_cannot_execute_or_advance_state(value):
    mon = UnexpiredPlanMonitor(
        Reference(), [Candidate("bad", lambda _: value, 0.1)], MonitorConfig(10)
    )
    mon.step(0)
    with pytest.raises(ValueError):
        mon.step(1)
    assert mon.call == 1 and mon.rung == 0


def test_returned_action_does_not_alias_stored_reference():
    mon = UnexpiredPlanMonitor(Reference(), [Candidate("fast", predictor, 0.2)], MonitorConfig(10))
    first = mon.step(0)
    first.executed[:] = 99
    assert mon.plan[0, 0] == 1


def test_history_is_bounded_by_control_calls():
    mon = UnexpiredPlanMonitor(
        Reference(),
        [Candidate("a", predictor, 0.1)],
        MonitorConfig(10, repetition_window=2),
    )
    for t in range(4):
        mon.step(t)
    assert all(t >= 2 for t, _ in mon.history)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon": 0},
        {"horizon": 10, "theta": float("nan")},
        {"horizon": 10, "epsilon": 0},
        {"horizon": 10, "repetition_window": 0},
        {"horizon": 10, "drift_by_age": {1: -0.1}},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        MonitorConfig(**kwargs)


def test_unsorted_ladder_rejected():
    with pytest.raises(ValueError):
        UnexpiredPlanMonitor(
            Reference(),
            [Candidate("a", predictor, 2), Candidate("b", predictor, 1)],
            MonitorConfig(10),
        )


def test_two_veto_reasons_advance_only_one_rung():
    reference = lambda _: np.ones((6, 1))
    candidates = [Candidate(str(i), lambda _: np.ones((6, 1)) * 2, i + 1) for i in range(3)]
    mon = UnexpiredPlanMonitor(
        reference,
        candidates,
        MonitorConfig(6, repetition_threshold=2.0),
    )
    mon.step(None)
    decision = mon.step(None)
    assert decision.veto_reasons == ("plan_deviation", "chunk_repetition")
    assert decision.rung_after == 1
    assert [t for t, _ in mon.history] == [0]  # A rejected chunk was never emitted.


def test_online_check_uses_only_executed_prefix_unlike_offline_selection():
    reference = lambda _: np.ones((4, 1))
    mon = UnexpiredPlanMonitor(
        reference,
        [Candidate("tail_error", lambda _: [[1], [1], [20], [20]], 1)],
        MonitorConfig(4),
    )
    mon.step(None)
    decision = mon.step(None)
    assert decision.source == "accelerated" and decision.delta_raw == 0


def test_exact_deviation_threshold_is_not_a_veto():
    mon = UnexpiredPlanMonitor(
        lambda _: np.ones((4, 1)),
        [Candidate("at_boundary", lambda _: np.ones((4, 1)) * 2, 1)],
        MonitorConfig(4, theta=0.5, epsilon=2),
    )
    mon.step(None)
    decision = mon.step(None)
    assert decision.delta_raw == 0.5 and not decision.veto_reasons


def test_empty_ladder_is_a_reference_only_controller():
    ref = Reference()
    monitor = UnexpiredPlanMonitor(ref, [], MonitorConfig(10))
    assert all(monitor.step(t).reference_forward for t in range(8))
    assert ref.calls == list(range(8))


def test_reference_refresh_failure_does_not_replace_a_valid_plan():
    def reference(t):
        return np.ones((4, 1)) if t == 0 else np.full((4, 1), np.nan)

    monitor = UnexpiredPlanMonitor(
        reference,
        [Candidate("valid", lambda _: np.ones((4, 1)), 1)],
        MonitorConfig(4),
    )
    monitor.step(0)
    monitor.step(1)
    with pytest.raises(ValueError):
        monitor.step(2)
    assert monitor.call == 2 and monitor.reference_call == 0
    np.testing.assert_array_equal(monitor.plan, np.ones((4, 1)))


def test_drift_age_cannot_refer_to_an_expired_reference():
    with pytest.raises(ValueError, match="ages 1 to m-1"):
        MonitorConfig(10, drift_by_age={5: 0.1})
