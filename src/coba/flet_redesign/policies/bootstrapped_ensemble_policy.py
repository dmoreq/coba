"""Bootstrapped ensemble policy."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

from coba.flet_redesign.contracts import BanditPolicy, DebugSnapshotProvider


class BootstrappedEnsemblePolicy(BanditPolicy[str, dict[str, Any]], DebugSnapshotProvider):
    """Ensemble of bootstrap heads with uncertainty from head disagreement."""

    def __init__(self, n_heads: int = 8, seed: int = 0) -> None:
        if n_heads < 2:
            raise ValueError("n_heads must be >= 2")
        self.n_heads = n_heads
        self._seed = seed
        self._rng = random.Random(seed)
        self._means: dict[str, list[float]] = {}
        self._counts: dict[str, list[int]] = {}
        self._last_scores: dict[str, float] = {}

    def reset(self) -> None:
        self._rng = random.Random(self._seed)
        self._means.clear()
        self._counts.clear()
        self._last_scores.clear()

    def select_arm(self, context: dict[str, Any], arms: Sequence[str]) -> str:
        _ = context
        if not arms:
            raise ValueError("BootstrappedEnsemblePolicy requires at least one arm")
        self._ensure_arms(arms)

        best_arm = None
        best_score = -float("inf")
        scores: dict[str, float] = {}
        for arm in arms:
            means = self._means[arm]
            avg = sum(means) / float(len(means))
            variance = sum((value - avg) ** 2 for value in means) / float(len(means))
            uncertainty = math.sqrt(max(0.0, variance))
            score = avg + uncertainty
            scores[arm] = score
            if score > best_score:
                best_score = score
                best_arm = arm
        self._last_scores = scores
        assert best_arm is not None
        return best_arm

    def update(self, context: dict[str, Any], arm: str, reward: float) -> None:
        _ = context
        self._ensure_arms([arm])
        for head in range(self.n_heads):
            if self._rng.random() < 0.7:
                count = self._counts[arm][head] + 1
                mean = self._means[arm][head]
                self._means[arm][head] = mean + (reward - mean) / float(count)
                self._counts[arm][head] = count

    def get_debug_snapshot(self) -> dict[str, Any]:
        return {
            "n_heads": self.n_heads,
            "scores": self._last_scores,
            "arms": {
                arm: {
                    "means": self._means[arm],
                    "counts": self._counts[arm],
                }
                for arm in self._means
            },
        }

    def _ensure_arms(self, arms: Sequence[str]) -> None:
        for arm in arms:
            if arm not in self._means:
                self._means[arm] = [0.0 for _ in range(self.n_heads)]
                self._counts[arm] = [0 for _ in range(self.n_heads)]
