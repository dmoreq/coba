"""Hybrid LinUCB policy with shared and arm-specific terms."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from coba.flet_redesign.contracts import BanditPolicy, DebugSnapshotProvider
from coba.flet_redesign.policies.contextual_utils import context_to_vector


class LinUCBHybridPolicy(BanditPolicy[str, dict[str, Any]], DebugSnapshotProvider):
    """Simplified hybrid LinUCB combining shared and arm-specific linear models."""

    def __init__(self, feature_order: Sequence[str], n_shared: int = 1, alpha: float = 1.0) -> None:
        if n_shared < 1:
            raise ValueError("n_shared must be >= 1")
        self.feature_order = tuple(feature_order)
        self.n_shared = min(n_shared, len(self.feature_order))
        self.alpha = alpha
        self._dim = len(self.feature_order)
        self._shared_theta = np.zeros(self.n_shared, dtype=float)
        self._arm_theta: dict[str, np.ndarray] = {}
        self._arm_count: dict[str, int] = {}
        self._last_scores: dict[str, float] = {}

    def reset(self) -> None:
        self._shared_theta = np.zeros(self.n_shared, dtype=float)
        self._arm_theta.clear()
        self._arm_count.clear()
        self._last_scores.clear()

    def select_arm(self, context: dict[str, Any], arms: Sequence[str]) -> str:
        if not arms:
            raise ValueError("LinUCBHybridPolicy requires at least one arm")
        x = np.array(context_to_vector(context, self.feature_order), dtype=float)
        z = x[: self.n_shared]
        tail = x[self.n_shared :]
        self._ensure_arms(arms)

        best_arm = None
        best_score = -float("inf")
        scores: dict[str, float] = {}
        for arm in arms:
            mean = float(self._shared_theta.T @ z) + float(self._arm_theta[arm].T @ tail)
            bonus = self.alpha / math.sqrt(float(self._arm_count[arm] + 1))
            score = mean + bonus
            scores[arm] = score
            if score > best_score:
                best_score = score
                best_arm = arm
        self._last_scores = scores
        assert best_arm is not None
        return best_arm

    def update(self, context: dict[str, Any], arm: str, reward: float) -> None:
        x = np.array(context_to_vector(context, self.feature_order), dtype=float)
        z = x[: self.n_shared]
        tail = x[self.n_shared :]
        self._ensure_arms([arm])
        eta = 1.0 / float(self._arm_count[arm] + 1)
        self._shared_theta += eta * reward * z
        self._arm_theta[arm] += eta * reward * tail
        self._arm_count[arm] += 1

    def get_debug_snapshot(self) -> dict[str, Any]:
        return {
            "n_shared": self.n_shared,
            "scores": self._last_scores,
            "shared_theta": self._shared_theta.tolist(),
            "arms": {
                arm: {
                    "theta": self._arm_theta[arm].tolist(),
                    "count": self._arm_count[arm],
                }
                for arm in self._arm_theta
            },
        }

    def _ensure_arms(self, arms: Sequence[str]) -> None:
        arm_dim = max(0, self._dim - self.n_shared)
        for arm in arms:
            if arm not in self._arm_theta:
                self._arm_theta[arm] = np.zeros(arm_dim, dtype=float)
                self._arm_count[arm] = 0
