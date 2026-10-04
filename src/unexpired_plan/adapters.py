"""Small explicit boundary between model action conventions and the monitor."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .metrics import action_array


@dataclass
class ActionNormalizer:
    center: np.ndarray
    scale: np.ndarray

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        self.scale = np.asarray(self.scale, dtype=float)
        if self.center.ndim != 1 or self.scale.shape != self.center.shape:
            raise ValueError(
                "One normalization parameter per action dimension required"
            )
        if (
            not np.isfinite(self.center).all()
            or not np.isfinite(self.scale).all()
            or (self.scale <= 0).any()
        ):
            raise ValueError("Finite centers and positive finite scales required")

    def normalize(self, action):
        a = action_array(action)
        if a.shape[1] != len(self.center):
            raise ValueError("Wrong action dimension")
        return (a - self.center) / self.scale

    def denormalize(self, action):
        a = action_array(action)
        if a.shape[1] != len(self.center):
            raise ValueError("Wrong action dimension")
        return a * self.scale + self.center


class NormalizedPolicy:
    """Adapter for a callable returning physical-coordinate NumPy actions.

    Model-specific observation preprocessing and checkpoint loading are explicit
    caller responsibilities. No hidden action slicing or gripper rescaling.
    """

    def __init__(self, predict, normalizer):
        self.predict = predict
        self.normalizer = normalizer

    def __call__(self, observation):
        return self.normalizer.normalize(self.predict(observation))

    def reset(self):
        if hasattr(self.predict, "reset"):
            self.predict.reset()
