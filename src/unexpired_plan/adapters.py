"""Small explicit boundary between model action conventions and the monitor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import action_array


@dataclass(frozen=True)
class ActionNormalizer:
    center: np.ndarray
    scale: np.ndarray

    def __post_init__(self):
        center = np.array(self.center, dtype=float, copy=True)
        scale = np.array(self.scale, dtype=float, copy=True)
        if center.ndim != 1 or not center.size or scale.shape != center.shape:
            raise ValueError("One normalization parameter per action dimension required")
        if not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Finite centers and positive finite scales required")
        center.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)

    def normalize(self, action):
        a = action_array(action)
        if a.shape[1] != len(self.center):
            raise ValueError("Wrong action dimension")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return action_array((a - self.center) / self.scale)

    def denormalize(self, action):
        a = action_array(action)
        if a.shape[1] != len(self.center):
            raise ValueError("Wrong action dimension")
        with np.errstate(over="ignore", invalid="ignore"):
            return action_array(a * self.scale + self.center)


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
