"""Random baseline policy for discrete arm selection."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from coba.flet_redesign.contracts import BanditPolicy


class RandomPolicy(BanditPolicy[Any, Any]):
    """Uniform random arm chooser with deterministic seeding."""

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)

    def reset(self) -> None:
        self._rng = random.Random(self._seed)

    def select_arm(self, context: Any, arms: Sequence[Any]) -> Any:
        _ = context
        if not arms:
            raise ValueError("RandomPolicy requires at least one arm")
        return self._rng.choice(list(arms))

    def update(self, context: Any, arm: Any, reward: float) -> None:
        _ = context, arm, reward
