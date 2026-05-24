"""Tree-style Thompson Sampling with context buckets."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from web.contracts import BanditPolicy, DebugSnapshotProvider


class TreeTSPolicy(BanditPolicy[str, dict[str, Any]], DebugSnapshotProvider):
    """Bucketized Beta-Bernoulli Thompson Sampling."""

    def __init__(self, context_key: str, seed: int = 0) -> None:
        self.context_key = context_key
        self._seed = seed
        self._rng = random.Random(seed)
        self._stats: dict[str, dict[int, tuple[float, float]]] = {}
        self._last_scores: dict[str, float] = {}

    def reset(self) -> None:
        self._rng = random.Random(self._seed)
        self._stats.clear()
        self._last_scores.clear()

    def select_arm(self, context: dict[str, Any], arms: Sequence[str]) -> str:
        if not arms:
            raise ValueError("TreeTSPolicy requires at least one arm")
        bucket = self._bucket(context)
        self._ensure_arms(arms)

        best_arm = None
        best_score = -float("inf")
        scores: dict[str, float] = {}
        for arm in arms:
            alpha, beta = self._stats[arm].get(bucket, (1.0, 1.0))
            score = self._rng.betavariate(alpha, beta)
            scores[arm] = score
            if score > best_score:
                best_score = score
                best_arm = arm
        self._last_scores = scores
        assert best_arm is not None
        return best_arm

    def update(self, context: dict[str, Any], arm: str, reward: float) -> None:
        bucket = self._bucket(context)
        self._ensure_arms([arm])
        alpha, beta = self._stats[arm].get(bucket, (1.0, 1.0))
        clipped = min(1.0, max(0.0, reward))
        self._stats[arm][bucket] = (alpha + clipped, beta + (1.0 - clipped))

    def get_debug_snapshot(self) -> dict[str, Any]:
        return {
            "context_key": self.context_key,
            "scores": self._last_scores,
            "arms": {
                arm: {
                    str(bucket): {"alpha": stat[0], "beta": stat[1]}
                    for bucket, stat in buckets.items()
                }
                for arm, buckets in self._stats.items()
            },
        }

    def _bucket(self, context: dict[str, Any]) -> int:
        value = context.get(self.context_key, 0.0)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int | float):
            return int(float(value) // 1)
        return hash(str(value)) % 10

    def _ensure_arms(self, arms: Sequence[str]) -> None:
        for arm in arms:
            if arm not in self._stats:
                self._stats[arm] = {}
