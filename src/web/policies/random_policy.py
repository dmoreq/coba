"""Random baseline policy for discrete arm selection."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from web.contracts import BanditPolicy, DebugSnapshotProvider


class RandomPolicy(BanditPolicy[Any, Any], DebugSnapshotProvider):
    """Uniform random arm chooser with deterministic seeding."""

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._counts: dict[str, int] = {}

    def reset(self) -> None:
        self._rng = random.Random(self._seed)
        self._counts.clear()

    def select_arm(self, context: Any, arms: Sequence[Any]) -> Any:
        _ = context
        if not arms:
            raise ValueError("RandomPolicy requires at least one arm")
        return self._rng.choice(list(arms))

    def update(self, context: Any, arm: Any, reward: float) -> None:
        _ = context, reward
        key = str(arm)
        self._counts[key] = self._counts.get(key, 0) + 1

    def get_debug_snapshot(self) -> dict[str, Any]:
        total = sum(self._counts.values()) or 1
        return {
            "policy": "random",
            "total_pulls": total,
            "arm_counts": dict(self._counts),
            "pull_distribution": {k: v / total for k, v in self._counts.items()},
        }
