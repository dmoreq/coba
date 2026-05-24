"""Lightweight GP-UCB style policy."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from coba.flet_redesign.contracts import BanditPolicy, DebugSnapshotProvider


class GPUCBPolicy(BanditPolicy[str, dict[str, Any]], DebugSnapshotProvider):
    """Approximate GP-UCB using per-arm running moments."""

    def __init__(self, beta: float = 1.5) -> None:
        if beta <= 0.0:
            raise ValueError("beta must be > 0")
        self.beta = beta
        self._count: dict[str, int] = {}
        self._mean: dict[str, float] = {}
        self._m2: dict[str, float] = {}
        self._last_scores: dict[str, float] = {}

    def reset(self) -> None:
        self._count.clear()
        self._mean.clear()
        self._m2.clear()
        self._last_scores.clear()

    def select_arm(self, context: dict[str, Any], arms: Sequence[str]) -> str:
        _ = context
        if not arms:
            raise ValueError("GPUCBPolicy requires at least one arm")
        self._ensure_arms(arms)
        best_arm = None
        best_score = -float("inf")
        scores: dict[str, float] = {}
        for arm in arms:
            n = self._count[arm]
            variance = self._variance(arm)
            uncertainty = math.sqrt(max(0.0, variance) / float(n + 1) + 1.0 / float(n + 1))
            score = self._mean[arm] + self.beta * uncertainty
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
        count = self._count[arm] + 1
        delta = reward - self._mean[arm]
        mean = self._mean[arm] + delta / float(count)
        delta2 = reward - mean
        self._m2[arm] += delta * delta2
        self._mean[arm] = mean
        self._count[arm] = count

    def get_debug_snapshot(self) -> dict[str, Any]:
        return {
            "beta": self.beta,
            "scores": self._last_scores,
            "arms": {
                arm: {
                    "count": self._count[arm],
                    "mean": self._mean[arm],
                    "variance": self._variance(arm),
                }
                for arm in self._count
            },
        }

    def _ensure_arms(self, arms: Sequence[str]) -> None:
        for arm in arms:
            if arm not in self._count:
                self._count[arm] = 0
                self._mean[arm] = 0.0
                self._m2[arm] = 0.0

    def _variance(self, arm: str) -> float:
        count = self._count[arm]
        if count <= 1:
            return 1.0
        return self._m2[arm] / float(count - 1)
