"""Explicit compute baselines; for future experiments, not new results."""

from __future__ import annotations
from collections import deque
import numpy as np
from .metrics import action_array, deviation, _horizons


class ScheduledReplay:
    """§V open-loop replay: Cc=0, reference period m, execute aligned slices."""

    def __init__(self, reference, horizon, execution_horizon=2):
        self.reference = reference
        self.horizon = horizon
        self.execution_horizon = execution_horizon
        self.m = _horizons(horizon, execution_horizon)
        self.reset()

    def reset(self):
        self.call = 0
        self.plan = None
        self.reference_forwards = 0

    def step(self, observation):
        j = self.call % self.m
        if j == 0:
            self.plan = action_array(self.reference(observation)).copy()
            if len(self.plan) != self.horizon:
                raise ValueError("Wrong reference horizon")
            self.reference_forwards += 1
        i = j * self.execution_horizon
        out = self.plan[i : i + self.execution_horizon].copy()
        self.call += 1
        return out


class ScheduleOnly:
    """Same reference schedule as the method, all free checks disabled."""

    def __init__(self, reference, candidate, horizon, execution_horizon=2):
        self.reference = reference
        self.candidate = candidate
        self.horizon = horizon
        self.h_exec = execution_horizon
        self.m = _horizons(horizon, execution_horizon)
        self.reset()

    def reset(self):
        self.call = 0
        self.reference_forwards = 0

    def step(self, observation):
        is_reference = self.call % self.m == 0
        p = self.reference if is_reference else self.candidate
        a = action_array(p(observation))
        if len(a) != self.horizon:
            raise ValueError("Wrong prediction horizon")
        self.reference_forwards += int(is_reference)
        self.call += 1
        return a[: self.h_exec].copy()


class PaidDeviation:
    """Recompute the reference on every call to read local deviation.

    Both predictors are charged when acceleration remains active. Unlike the
    main method, fallback uses a fresh observation. Response is configurable.
    """

    def __init__(
        self, reference, candidate, theta=0.15, execution_horizon=2, absorbing=True
    ):
        _horizons(execution_horizon, execution_horizon)
        if theta <= 0 or not np.isfinite(theta):
            raise ValueError("Positive theta required")
        self.reference = reference
        self.candidate = candidate
        self.theta = theta
        self.h_exec = execution_horizon
        self.absorbing = absorbing
        self.reset()

    def reset(self):
        self.retired = False
        self.reference_forwards = 0
        self.candidate_forwards = 0

    def step(self, observation):
        r = action_array(self.reference(observation))
        self.reference_forwards += 1
        if len(r) < self.h_exec:
            raise ValueError("Reference does not cover execution")
        if self.retired:
            return r[: self.h_exec].copy()
        a = action_array(self.candidate(observation))
        self.candidate_forwards += 1
        veto = deviation(a[: self.h_exec], r[: self.h_exec]) > self.theta
        self.retired = bool(veto and self.absorbing)
        return (r if veto else a)[: self.h_exec].copy()


class RateMatchedCoin:
    """Matched-rate admission baseline with explicit re-arming/absorbing response.

    A retirement-matched coin is a distinct experiment and is not approximated
    by pretending its admission rate is known.
    """

    def __init__(
        self,
        reference,
        candidate,
        acceptance_rate,
        seed,
        execution_horizon=2,
        absorbing=False,
    ):
        _horizons(execution_horizon, execution_horizon)
        if not 0 <= acceptance_rate <= 1:
            raise ValueError("Invalid acceptance rate")
        self.reference = reference
        self.candidate = candidate
        self.rate = acceptance_rate
        self.seed = seed
        self.h_exec = execution_horizon
        self.absorbing = absorbing
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.retired = False

    def step(self, observation):
        accepts = not self.retired and self.rng.random() < self.rate
        if not accepts and self.absorbing:
            self.retired = True
        a = action_array((self.candidate if accepts else self.reference)(observation))
        if len(a) < self.h_exec:
            raise ValueError("Predictor does not cover execution")
        return a[: self.h_exec].copy()
