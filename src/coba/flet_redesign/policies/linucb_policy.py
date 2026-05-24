"""Contextual LinUCB policy."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from coba.flet_redesign.contracts import BanditPolicy, DebugSnapshotProvider
from coba.flet_redesign.policies.contextual_utils import context_to_vector


class LinUCBPolicy(BanditPolicy[str, dict[str, Any]], DebugSnapshotProvider):
    """Per-arm linear upper confidence bound policy."""

    def __init__(
        self, feature_order: Sequence[str], alpha: float = 1.0, l2_lambda: float = 1.0
    ) -> None:
        if alpha <= 0.0:
            raise ValueError("alpha must be > 0")
        if l2_lambda <= 0.0:
            raise ValueError("l2_lambda must be > 0")
        self.feature_order = tuple(feature_order)
        self.alpha = alpha
        self.l2_lambda = l2_lambda
        self._dim = len(self.feature_order)
        self._a: dict[str, np.ndarray] = {}
        self._b: dict[str, np.ndarray] = {}
        self._last_scores: dict[str, float] = {}

    def reset(self) -> None:
        self._a.clear()
        self._b.clear()
        self._last_scores.clear()

    def select_arm(self, context: dict[str, Any], arms: Sequence[str]) -> str:
        if not arms:
            raise ValueError("LinUCBPolicy requires at least one arm")
        x = np.array(context_to_vector(context, self.feature_order), dtype=float)
        self._ensure_arms(list(arms))

        best_arm = None
        best_score = -float("inf")
        scores: dict[str, float] = {}
        for arm in arms:
            a_inv = np.linalg.inv(self._a[arm])
            theta = a_inv @ self._b[arm]
            exploit = float(theta.T @ x)
            explore = self.alpha * math.sqrt(float(x.T @ a_inv @ x))
            score = exploit + explore
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
        self._a[arm] += np.outer(x, x)
        self._b[arm] += reward * x

    def get_debug_snapshot(self) -> dict[str, Any]:
        return {
            "feature_order": self.feature_order,
            "scores": self._last_scores,
            "arms": {
                arm: {
                    "a": self._a[arm].tolist(),
                    "b": self._b[arm].tolist(),
                }
                for arm in self._a
            },
        }

    def _ensure_arms(self, arms: Sequence[str]) -> None:
        identity = np.eye(self._dim, dtype=float) * self.l2_lambda
        zeros = np.zeros(self._dim, dtype=float)
        for arm in arms:
            if arm not in self._a:
                self._a[arm] = identity.copy()
                self._b[arm] = zeros.copy()
