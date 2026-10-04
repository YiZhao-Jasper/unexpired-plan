"""Independent implementation of the unexpired-plan monitor.

This package does not contain the manuscript's original checkpoints or data.
"""

from .core import Candidate, Decision, MonitorConfig, UnexpiredPlanMonitor
from .metrics import deviation, effective_speedup, expected_damage, coverage

__all__ = [
    "Candidate",
    "Decision",
    "MonitorConfig",
    "UnexpiredPlanMonitor",
    "deviation",
    "effective_speedup",
    "expected_damage",
    "coverage",
]
