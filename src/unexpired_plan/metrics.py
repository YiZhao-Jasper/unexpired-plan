"""Equations (1), (3), (4), (5), (6); no benchmark numbers are computed here."""

from __future__ import annotations
import math
import numpy as np


def action_array(value):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or not a.size or not np.isfinite(a).all():
        raise ValueError(
            "Actions must be a finite, nonempty [horizon, dimension] array"
        )
    return a


def deviation(action, reference, epsilon=1e-8):
    """Eq. (1), in an adapter's normalized action coordinates."""
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    a, r = action_array(action), action_array(reference)
    if a.shape != r.shape:
        raise ValueError("Compared actions must have identical shapes and time indices")
    # Scale before subtraction and summation to avoid overflow on large finite
    # inputs. This is algebraically the same reference-normalized L1 ratio.
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(r))), epsilon)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        value = float(
            np.abs(a / scale - r / scale).sum()
            / (np.abs(r / scale).sum() + epsilon / scale)
        )
    if not math.isfinite(value):
        raise ValueError("Normalized deviation is not finite")
    return value


def _horizons(horizon, execution_horizon):
    if any(
        isinstance(x, bool) or not isinstance(x, int)
        for x in (horizon, execution_horizon)
    ):
        raise ValueError("Horizons must be integers")
    if execution_horizon < 1 or horizon < execution_horizon:
        raise ValueError("Require horizon >= execution_horizon >= 1")
    return horizon // execution_horizon


def coverage(horizon, execution_horizon, reference_period=None):
    """Distinguish free coverage, guarded all-call share, and accelerated coverage."""
    m = _horizons(horizon, execution_horizon)
    k = m if reference_period is None else reference_period
    if isinstance(k, bool) or not isinstance(k, int) or k < m:
        raise ValueError("This frontier is defined at reference period k >= m")
    return {
        "m": m,
        "period": k,
        "free_all_calls": (m - 1) / k,
        "guarded_all_calls": m / k,
        "checked_accelerated_calls": (m - 1) / (k - 1) if k > 1 else None,
    }


def effective_speedup(m, raw_speedup):
    """Eq. (3), ideal fixed-rung mean compute; not end-to-end task latency."""
    if (
        isinstance(m, bool)
        or not isinstance(m, int)
        or m < 1
        or math.isnan(raw_speedup)
        or raw_speedup <= 0
    ):
        raise ValueError("m >= 1 and positive raw speedup required")
    return m / (1 + (m - 1) / raw_speedup)


def expected_damage(m, k, calls, damage_probability, false_negative_rate):
    """Eq. (4); expected chunks, NOT a probability or a guarantee."""
    coverage(m, 1, k)
    if (
        not math.isfinite(calls)
        or calls < 0
        or not 0 <= damage_probability <= 1
        or not 0 <= false_negative_rate <= 1
    ):
        raise ValueError("Invalid exposure inputs")
    return (
        damage_probability * calls * ((k - m) / k + (m - 1) / k * false_negative_rate)
    )


def exposure_step_ratio(m, false_negative_rate):
    if m <= 1 or not 0 < false_negative_rate <= 1:
        raise ValueError("Ratio requires m > 1 and a nonzero miss rate")
    f = false_negative_rate
    return m * (1 + (m - 1) * f) / ((m + 1) * (m - 1) * f)


def composite_miss_rate(weights, delta_misses, repetition_misses):
    """Eq. (6) plus Fréchet bounds; independence is an explicit assumption."""
    w, a, b = [
        np.asarray(x, dtype=float) for x in (weights, delta_misses, repetition_misses)
    ]
    if w.ndim != 1 or not w.size or a.shape != w.shape or b.shape != w.shape:
        raise ValueError("Mechanism vectors must have equal nonzero length")
    if not all(np.isfinite(x).all() for x in (w, a, b)) or not np.isclose(w.sum(), 1):
        raise ValueError("Finite weights summing to one required")
    if any(((x < 0) | (x > 1)).any() for x in (w, a, b)):
        raise ValueError("Rates and weights must lie in [0,1]")
    return {
        "independent": float(w @ (a * b)),
        "lower": float(w @ np.maximum(0, a + b - 1)),
        "upper": float(w @ np.minimum(a, b)),
    }
