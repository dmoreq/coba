"""Tree-style UCB policy using context bucketization."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from web.contracts import BanditPolicy, DebugSnapshotProvider


class TreeUCBPolicy(BanditPolicy[str, dict[str, Any]], DebugSnapshotProvider):
    """Bucketized context policy approximating tree UCB behavior."""

    def __init__(self, context_key: str, alpha: float = 0.8) -> None:
        if alpha <= 0.0:
            raise ValueError("alpha must be > 0")
        self.context_key = context_key
        self.alpha = alpha
        self._stats: dict[str, dict[int, tuple[int, float]]] = {}
        self._last_scores: dict[str, float] = {}

    def reset(self) -> None:
        self._stats.clear()
        self._last_scores.clear()

    def select_arm(self, context: dict[str, Any], arms: Sequence[str]) -> str:
        if not arms:
            raise ValueError("TreeUCBPolicy requires at least one arm")
        bucket = self._bucket(context)
        self._ensure_arms(arms)

        best_arm = None
        best_score = -float("inf")
        scores: dict[str, float] = {}
        for arm in arms:
            pulls, mean = self._stats[arm].get(bucket, (0, 0.0))
            score = mean + self.alpha / math.sqrt(float(pulls + 1))
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
        pulls, mean = self._stats[arm].get(bucket, (0, 0.0))
        next_pulls = pulls + 1
        next_mean = mean + (reward - mean) / float(next_pulls)
        self._stats[arm][bucket] = (next_pulls, next_mean)

    def get_debug_snapshot(self) -> dict[str, Any]:
        return {
            "context_key": self.context_key,
            "scores": self._last_scores,
            "arms": {
                arm: {
                    str(bucket): {"pulls": stat[0], "mean": stat[1]}
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
