"""Linear Thompson Sampling (LinTS) policy."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from web.contracts import BanditPolicy, DebugSnapshotProvider
from web.policies.contextual_utils import context_to_vector


class LinTSPolicy(BanditPolicy[str, dict[str, Any]], DebugSnapshotProvider):
    """Per-arm linear Thompson Sampling with posterior sampling."""

    def __init__(
        self,
        feature_order: Sequence[str],
        prior_variance: float = 1.0,
        l2_lambda: float = 1.0,
        seed: int = 0,
    ) -> None:
        if prior_variance <= 0.0:
            raise ValueError("prior_variance must be > 0")
        if l2_lambda <= 0.0:
            raise ValueError("l2_lambda must be > 0")
        self.feature_order = tuple(feature_order)
        self.prior_variance = prior_variance
        self.l2_lambda = l2_lambda
        self._dim = len(self.feature_order)
        self._rng = np.random.RandomState(seed)
        self._a: dict[str, np.ndarray] = {}
        self._b: dict[str, np.ndarray] = {}
        self._noise_variance: float = 1.0
        self._total_reward: float = 0.0
        self._total_sq_reward: float = 0.0
        self._n_updates: int = 0
        self._last_scores: dict[str, float] = {}
        self._last_theta: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self._a.clear()
        self._b.clear()
        self._last_scores.clear()
        self._last_theta.clear()
        self._noise_variance = 1.0
        self._total_reward = 0.0
        self._total_sq_reward = 0.0
        self._n_updates = 0

    def select_arm(self, context: dict[str, Any], arms: Sequence[str]) -> str:
        if not arms:
            raise ValueError("LinTSPolicy requires at least one arm")
        x = np.array(context_to_vector(context, self.feature_order), dtype=float)
        self._ensure_arms(list(arms))
        est_var = self._estimated_noise_variance()

        best_arm = None
        best_score = -float("inf")
        scores: dict[str, float] = {}
        thetas: dict[str, np.ndarray] = {}
        for arm in arms:
            a_mat = self._a[arm]
            a_inv = np.linalg.inv(a_mat)
            theta_hat = a_inv @ self._b[arm]
            cov = est_var * a_inv
            jitter = 1e-8 * np.eye(self._dim, dtype=float)
            cov += jitter
            try:
                theta_sample = self._rng.multivariate_normal(mean=theta_hat, cov=cov)
            except np.linalg.LinAlgError:
                theta_sample = theta_hat + self._rng.normal(0.0, math.sqrt(est_var), size=self._dim)

            score = float(np.dot(theta_sample, x))
            scores[arm] = score
            thetas[arm] = theta_sample
            if score > best_score:
                best_score = score
                best_arm = arm

        self._last_scores = scores
        self._last_theta = thetas
        assert best_arm is not None
        return best_arm

    def update(self, context: dict[str, Any], arm: str, reward: float) -> None:
        x = np.array(context_to_vector(context, self.feature_order), dtype=float)
        self._ensure_arms([arm])
        self._a[arm] += np.outer(x, x)
        self._b[arm] += reward * x
        self._total_reward += reward
        self._total_sq_reward += reward * reward
        self._n_updates += 1
        self._noise_variance = self._estimated_noise_variance()

    def get_debug_snapshot(self) -> dict[str, Any]:
        return {
            "feature_order": self.feature_order,
            "prior_variance": self.prior_variance,
            "estimated_noise_variance": self._noise_variance,
            "scores": self._last_scores,
            "arms": {
                arm: {
                    "a": self._a[arm].tolist() if arm in self._a else None,
                    "b": self._b[arm].tolist() if arm in self._b else None,
                    "theta_hat": (
                        (np.linalg.inv(self._a[arm]) @ self._b[arm]).tolist()
                        if arm in self._a
                        else None
                    ),
                    "theta_sample": (
                        self._last_theta[arm].tolist() if arm in self._last_theta else None
                    ),
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

    def _estimated_noise_variance(self) -> float:
        if self._n_updates <= 1:
            return self.prior_variance
        mean_reward = self._total_reward / float(self._n_updates)
        var = max(
            0.0,
            self._total_sq_reward / float(self._n_updates) - mean_reward * mean_reward,
        )
        return max(1e-6, var)
