"""Simple logistic-UCB style contextual policy."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from web.contracts import BanditPolicy, DebugSnapshotProvider
from web.policies.contextual_utils import context_to_vector


class LogisticUCBPolicy(BanditPolicy[str, dict[str, Any]], DebugSnapshotProvider):
    """Online logistic model with UCB-style exploration bonus."""

    def __init__(
        self,
        feature_order: Sequence[str],
        alpha: float = 0.5,
        learning_rate: float = 0.1,
    ) -> None:
        if alpha <= 0.0:
            raise ValueError("alpha must be > 0")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        self.feature_order = tuple(feature_order)
        self.alpha = alpha
        self.learning_rate = learning_rate
        self._dim = len(self.feature_order)
        self._theta: dict[str, np.ndarray] = {}
        self._pulls: dict[str, int] = {}
        self._last_scores: dict[str, float] = {}

    def reset(self) -> None:
        self._theta.clear()
        self._pulls.clear()
        self._last_scores.clear()

    def select_arm(self, context: dict[str, Any], arms: Sequence[str]) -> str:
        if not arms:
            raise ValueError("LogisticUCBPolicy requires at least one arm")
        x = np.array(context_to_vector(context, self.feature_order), dtype=float)
        self._ensure_arms(arms)

        best_arm = None
        best_score = -float("inf")
        scores: dict[str, float] = {}
        for arm in arms:
            mean = self._sigmoid(float(self._theta[arm].T @ x))
            explore = self.alpha / math.sqrt(float(self._pulls[arm] + 1))
            score = mean + explore
            scores[arm] = score
            if score > best_score:
                best_score = score
                best_arm = arm
        self._last_scores = scores
        assert best_arm is not None
        return best_arm

    def update(self, context: dict[str, Any], arm: str, reward: float) -> None:
        x = np.array(context_to_vector(context, self.feature_order), dtype=float)
        self._ensure_arms([arm])
        prob = self._sigmoid(float(self._theta[arm].T @ x))
        gradient = (reward - prob) * x
        self._theta[arm] += self.learning_rate * gradient
        self._pulls[arm] += 1

    def get_debug_snapshot(self) -> dict[str, Any]:
        return {
            "feature_order": self.feature_order,
            "scores": self._last_scores,
            "arms": {
                arm: {
                    "theta": self._theta[arm].tolist(),
                    "pulls": self._pulls[arm],
                }
                for arm in self._theta
            },
        }

    def _ensure_arms(self, arms: Sequence[str]) -> None:
        for arm in arms:
            if arm not in self._theta:
                self._theta[arm] = np.zeros(self._dim, dtype=float)
                self._pulls[arm] = 0

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            exp_neg = math.exp(-value)
            return 1.0 / (1.0 + exp_neg)
        exp_pos = math.exp(value)
        return exp_pos / (1.0 + exp_pos)
