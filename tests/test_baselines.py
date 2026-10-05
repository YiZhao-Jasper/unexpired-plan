import numpy as np
import pytest

from unexpired_plan.adapters import ActionNormalizer
from unexpired_plan.baselines import (
    PaidDeviation,
    RateMatchedCoin,
    ScheduledReplay,
    ScheduleOnly,
)


def test_scheduled_replay_executes_same_time_indices():
    calls = []

    def ref(t):
        calls.append(t)
        return np.arange(t * 2, t * 2 + 10).reshape(10, 1)

    b = ScheduledReplay(ref, 10)
    actual = np.concatenate([b.step(t) for t in range(10)]).ravel()
    np.testing.assert_array_equal(actual, np.arange(20))
    assert calls == [0, 5]


def test_schedule_only_charges_reference_at_correct_calls():
    b = ScheduleOnly(lambda _: np.ones((4, 1)), lambda _: np.zeros((4, 1)), 4)
    out = [b.step(t)[0, 0] for t in range(6)]
    assert out == [1, 0, 1, 0, 1, 0] and b.reference_forwards == 3


def test_paid_oracle_rearming_and_absorbing_diverge_after_same_veto():
    ref = lambda _: np.ones((2, 1))
    cand = lambda t: np.ones((2, 1)) * (2 if t == 0 else 1.01)
    absorb = PaidDeviation(ref, cand, absorbing=True)
    rearm = PaidDeviation(ref, cand, absorbing=False)
    absorb.step(0)
    rearm.step(0)
    assert absorb.step(1)[0, 0] == 1
    assert rearm.step(1)[0, 0] == 1.01
    assert absorb.candidate_forwards == 1 and rearm.candidate_forwards == 2


def test_zero_admission_coin_never_uses_candidate():
    b = RateMatchedCoin(
        lambda _: np.ones((2, 1)), lambda _: np.zeros((2, 1)), 0, seed=1, absorbing=True
    )
    assert all(b.step(t)[0, 0] == 1 for t in range(4))


def test_action_normalization_roundtrip():
    n = ActionNormalizer([1, 2], [2, 0.5])
    a = np.array([[5, 3], [2, 1]])
    np.testing.assert_allclose(n.denormalize(n.normalize(a)), a)


def test_paid_comparison_cannot_be_overwritten_by_a_shared_output_buffer():
    buffer = np.zeros((4, 1))

    def reference(_):
        buffer[:] = 1
        return buffer

    def candidate(_):
        buffer[:] = 2
        return buffer

    baseline = PaidDeviation(reference, candidate)
    np.testing.assert_array_equal(baseline.step(None), np.ones((2, 1)))
    assert baseline.retired
    assert baseline.reference_forwards == baseline.candidate_forwards == 1


def test_normalization_copies_parameters_and_rejects_nonfinite_outputs():
    center, scale = np.zeros(1), np.ones(1)
    normalizer = ActionNormalizer(center, scale)
    center[:] = 20
    scale[:] = 2
    np.testing.assert_array_equal(normalizer.normalize([[1]]), [[1]])
    with pytest.raises(ValueError):
        normalizer.scale[0] = 0
    with pytest.raises(ValueError):
        ActionNormalizer([0], [1e-308]).normalize([[1e308]])
    with pytest.raises(ValueError):
        ActionNormalizer([0], [1e308]).denormalize([[1e308]])
    with pytest.raises(ValueError):
        ActionNormalizer([], [])
