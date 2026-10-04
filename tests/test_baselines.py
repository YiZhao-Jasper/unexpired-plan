import numpy as np
from unexpired_plan.baselines import (
    ScheduledReplay,
    ScheduleOnly,
    PaidDeviation,
    RateMatchedCoin,
)
from unexpired_plan.adapters import ActionNormalizer


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
