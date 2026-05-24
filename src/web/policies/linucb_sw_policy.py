"""Sliding-window LinUCB policy."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np

from web.contracts import BanditPolicy, DebugSnapshotProvider
from web.policies.contextual_utils import context_to_vector


class LinUCBSWPolicy(BanditPolicy[str, dict[str, Any]], DebugSnapshotProvider):
    """LinUCB with per-arm sliding window updates."""

    def __init__(
        self,
        feature_order: Sequence[str],
        window_size: int = 200,
        alpha: float = 1.0,
        l2_lambda: float = 1.0,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if alpha <= 0.0:
            raise ValueError("alpha must be > 0")
        self.feature_order = tuple(feature_order)
        self.window_size = window_size
        self.alpha = alpha
        self.l2_lambda = l2_lambda
        self._dim = len(self.feature_order)
        self._buffers: dict[str, deque[tuple[np.ndarray, float]]] = {}
        self._a: dict[str, np.ndarray] = {}
        self._b: dict[str, np.ndarray] = {}
        self._last_scores: dict[str, float] = {}

    def reset(self) -> None:
        self._buffers.clear()
        self._a.clear()
        self._b.clear()
        self._last_scores.clear()

    def select_arm(self, context: dict[str, Any], arms: Sequence[str]) -> str:
        if not arms:
            raise ValueError("LinUCBSWPolicy requires at least one arm")
        x = np.array(context_to_vector(context, self.feature_order), dtype=float)
        self._ensure_arms(arms)

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
        buffer = self._buffers[arm]
        buffer.append((x, reward))
        if len(buffer) > self.window_size:
            buffer.popleft()
        self._rebuild_arm_matrices(arm)

    def get_debug_snapshot(self) -> dict[str, Any]:
        return {
            "feature_order": self.feature_order,
            "window_size": self.window_size,
            "scores": self._last_scores,
            "arms": {
                arm: {
                    "count": len(self._buffers[arm]),
                    "a": self._a[arm].tolist(),
                    "b": self._b[arm].tolist(),
                }
                for arm in self._a
            },
        }

    def _ensure_arms(self, arms: Sequence[str]) -> None:
        for arm in arms:
            if arm not in self._buffers:
                self._buffers[arm] = deque()
                self._rebuild_arm_matrices(arm)

    def _rebuild_arm_matrices(self, arm: str) -> None:
        a = np.eye(self._dim, dtype=float) * self.l2_lambda
        b = np.zeros(self._dim, dtype=float)
        for x, reward in self._buffers[arm]:
            a += np.outer(x, x)
            b += reward * x
        self._a[arm] = a
        self._b[arm] = b
