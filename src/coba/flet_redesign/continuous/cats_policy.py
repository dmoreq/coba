"""CATS-like policy for continuous actions."""

from __future__ import annotations

import random
from dataclasses import dataclass

from coba.flet_redesign.continuous.schemas import ContinuousActionSpace


@dataclass
class CATSLikePolicy:
    """Tree-sampling inspired continuous-action policy."""

    action_space: ContinuousActionSpace
    exploration: float = 0.25
    seed: int = 0

    def __post_init__(self) -> None:
        if self.exploration <= 0.0:
            raise ValueError("exploration must be > 0")
        self._rng = random.Random(self.seed)
        self._best_action = (self.action_space.min_value + self.action_space.max_value) / 2.0
        self._best_reward = float("-inf")
        self._history: list[tuple[float, float]] = []
        self._last_sampled: float | None = None

    def reset(self) -> None:
        self._rng = random.Random(self.seed)
        self._best_action = (self.action_space.min_value + self.action_space.max_value) / 2.0
        self._best_reward = float("-inf")
        self._history.clear()
        self._last_sampled = None

    def select_action(self) -> float:
        range_width = self.action_space.max_value - self.action_space.min_value
        std = max(1e-6, self.exploration * range_width / (1.0 + len(self._history) / 50.0))
        sampled = self._rng.gauss(self._best_action, std)
        action = self.action_space.clip(sampled)
        self._last_sampled = action
        return action

    def update(self, action: float, reward: float) -> None:
        self._history.append((action, reward))
        if reward > self._best_reward:
            self._best_reward = reward
            self._best_action = action

    def get_debug_snapshot(self) -> dict[str, float | int]:
        return {
            "best_action": self._best_action,
            "best_reward": self._best_reward,
            "history_size": len(self._history),
            "last_sampled_action": -1.0 if self._last_sampled is None else self._last_sampled,
        }
