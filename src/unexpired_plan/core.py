"""Full-coverage, veto-only controller (manuscript §IV).

The reference is called every floor(H / H_exec) control calls. On a veto,
the saved, time-aligned reference slice executes without a new forward pass.
The ladder can only move toward slower candidates within an episode.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Callable, Mapping, Optional, Sequence
from types import MappingProxyType
import time
import numpy as np
from .metrics import action_array, deviation, _horizons


@dataclass(frozen=True)
class Candidate:
    name: str
    predict: Callable
    cost: float
    mechanism: str = "recompute"

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Candidate name must be a nonempty string")
        if not callable(self.predict):
            raise ValueError("Candidate predictor must be callable")
        if not np.isfinite(self.cost) or self.cost <= 0:
            raise ValueError("Measured candidate cost must be finite and positive")
        if self.mechanism not in ("recompute", "reuse", "blend"):
            raise ValueError("Unknown mechanism")


@dataclass(frozen=True)
class MonitorConfig:
    horizon: int
    execution_horizon: int = 2
    theta: float = 0.15
    # Must be calibrated on reference-against-reference replay, not hand-picked.
    repetition_threshold: float = 0.0
    repetition_window: int = 8
    epsilon: float = 1e-8
    drift_by_age: Mapping[int, float] = field(default_factory=dict)

    def __post_init__(self):
        _horizons(self.horizon, self.execution_horizon)
        if not np.isfinite(self.theta) or self.theta <= 0:
            raise ValueError("theta must be finite and positive")
        if not np.isfinite(self.repetition_threshold) or self.repetition_threshold < 0:
            raise ValueError("repetition_threshold must be finite and nonnegative")
        if (
            isinstance(self.repetition_window, bool)
            or not isinstance(self.repetition_window, int)
            or self.repetition_window < 1
        ):
            raise ValueError("repetition_window must be a positive integer")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if any(
            isinstance(k, bool)
            or not isinstance(k, int)
            or k < 1
            or not np.isfinite(v)
            or v < 0
            for k, v in self.drift_by_age.items()
        ):
            raise ValueError(
                "Drift corrections must be nonnegative, keyed by positive age"
            )
        object.__setattr__(
            self, "drift_by_age", MappingProxyType(dict(self.drift_by_age))
        )


@dataclass
class Decision:
    call: int
    source: str
    executed: np.ndarray
    candidate: Optional[str]
    rung_before: int
    rung_after: int
    reference_age: int
    delta_raw: Optional[float] = None
    delta_corrected: Optional[float] = None
    repetition: Optional[float] = None
    veto_reasons: tuple = ()
    reference_forward: bool = False
    elapsed_seconds: float = 0.0

    def as_dict(self):
        out = dict(vars(self))
        out["executed"] = self.executed.tolist()
        return out


class UnexpiredPlanMonitor:
    def __init__(
        self, reference, candidates: Sequence[Candidate], config: MonitorConfig
    ):
        self.reference, self.candidates, self.config = (
            reference,
            tuple(candidates),
            config,
        )
        if not callable(reference):
            raise ValueError("Reference predictor must be callable")
        if len({c.name for c in self.candidates}) != len(self.candidates):
            raise ValueError("Candidate names must be unique")
        if any(a.cost > b.cost for a, b in zip(self.candidates, self.candidates[1:])):
            raise ValueError(
                "Ladder must be ordered from cheap to conservative by measured cost"
            )
        self.m = _horizons(config.horizon, config.execution_horizon)
        self.reset()

    def reset(self):
        """Only an episode boundary may re-arm the ladder."""
        self.call, self.rung = 0, 0
        self.plan, self.reference_call = None, -1
        self.history = deque(maxlen=self.config.repetition_window)
        self.action_dimension = None

    def _plan(self, value):
        a = action_array(value)
        if a.shape[0] != self.config.horizon:
            raise ValueError("Predictor must return exactly H normalized actions")
        if self.action_dimension is not None and a.shape[1] != self.action_dimension:
            raise ValueError("Action dimension changed during the episode")
        return a.copy()

    def step(self, observation):
        start = time.perf_counter()
        c, t, before = self.config, self.call, self.rung
        # Prepare the window without changing state until prediction succeeds.
        recent = deque(
            ((j, p) for j, p in self.history if j >= t - c.repetition_window),
            maxlen=c.repetition_window,
        )
        terminal = self.rung == len(self.candidates)
        # After ladder exhaustion the reference policy owns every call.
        if t % self.m == 0 or terminal:
            p = self._plan(self.reference(observation))
            self.plan, self.reference_call = p, t
            self.action_dimension = p.shape[1]
            out = Decision(
                t,
                "reference",
                p[: c.execution_horizon].copy(),
                None,
                before,
                self.rung,
                0,
                reference_forward=True,
            )
            recent.append((t, p.copy()))
        else:
            age = t - self.reference_call
            offset = age * c.execution_horizon
            r = self.plan[offset : offset + c.execution_horizon]
            if r.shape[0] != c.execution_horizon:
                raise RuntimeError("Expired plan cannot guard or supply a fallback")
            candidate = self.candidates[self.rung]
            # Invalid proposals fail loudly. They must not silently execute.
            a = self._plan(candidate.predict(observation))
            raw = deviation(a[: c.execution_horizon], r, c.epsilon)
            correction = (
                c.drift_by_age.get(age, 0.0)
                if candidate.mechanism == "recompute"
                else 0.0
            )
            corrected = max(0.0, raw - correction)
            rep = min((deviation(a, h, c.epsilon) for _, h in recent), default=None)
            reasons = []
            if corrected > c.theta:
                reasons.append("plan_deviation")
            if rep is not None and rep < c.repetition_threshold:
                reasons.append("chunk_repetition")
            if reasons:
                self.rung += 1
                out = Decision(
                    t,
                    "unexpired_fallback",
                    r.copy(),
                    candidate.name,
                    before,
                    self.rung,
                    age,
                    raw,
                    corrected,
                    rep,
                    tuple(reasons),
                )
                # The fallback is only H_exec rows. Do not fabricate a full plan
                # by padding it. Retain full plans from the previous K calls.
            else:
                out = Decision(
                    t,
                    "accelerated",
                    a[: c.execution_horizon].copy(),
                    candidate.name,
                    before,
                    self.rung,
                    age,
                    raw,
                    corrected,
                    rep,
                )
                recent.append((t, a.copy()))
        self.history = recent
        self.call += 1
        out.elapsed_seconds = time.perf_counter() - start
        return out
