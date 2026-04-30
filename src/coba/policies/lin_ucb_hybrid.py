"""
LinUCB-Hybrid arm model.

Reference:
  Li et al., "A contextual-bandit approach to personalized news", WWW 2010.
  Section 3.2 — Hybrid Linear Bandits.

Context split:
  The full context vector x (length n_features = n_shared + n_arm) is split into:
    z = x[:n_shared]   — shared features (e.g. user demographics, time-of-day)
    x_arm = x[n_shared:]  — arm-specific features (e.g. item embeddings)

Score formula (approximated from Li et al. Eq. 7–9):
  E[y | z, x_arm] = z @ β_shared + x_arm @ θ_arm
  UCB(z, x_arm)   = alpha * sqrt(z @ A0_inv @ z + x_arm @ A_arm_inv @ x_arm)

The shared β_shared is trained on ALL (arm, reward) observations so it benefits
from cross-arm data transfer. θ_arm is trained only on observations from this arm.

This is an approximation of the exact hybrid UCB (which includes a cross-covariance
term), but it captures the key property: shared features learn faster because they
pool data across arms.

Usage:
  Create one SharedRidge per cluster (not per arm) and pass the same instance to
  every LinUCBHybridArmModel in that cluster. The SharedRidge is updated on every
  arm pull with the shared feature slice.
"""

from __future__ import annotations

import numpy as np

from coba.policies.base import BaseArmModel
from coba.policies.ridge import RidgeRegression
from coba.types import Arm


class LinUCBHybridArmModel(BaseArmModel):
    """Per-arm LinUCB-Hybrid model with cross-arm shared feature learning.

    Args:
        arm: Arm identifier.
        n_shared: Number of shared context dimensions (first n_shared elements of x).
        n_arm: Number of arm-specific context dimensions (remaining elements of x).
        shared_ridge: A RidgeRegression instance shared across all arms in this cluster.
                      Updated on every arm pull; provides cross-arm feature learning.
        alpha: UCB exploration width.
        l2_lambda: L2 regularization for the per-arm ridge.
        gamma: Discount factor for non-stationarity.
        rng: NumPy random generator.
    """

    def __init__(
        self,
        arm: Arm,
        n_shared: int,
        n_arm: int,
        shared_ridge: RidgeRegression,
        alpha: float = 1.0,
        l2_lambda: float = 1.0,
        gamma: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if n_shared < 0:
            raise ValueError(f"n_shared must be >= 0, got {n_shared}")
        if n_arm < 0:
            raise ValueError(f"n_arm must be >= 0, got {n_arm}")
        if n_shared + n_arm == 0:
            raise ValueError("n_shared + n_arm must be > 0")

        super().__init__(arm, rng or np.random.default_rng())
        self.n_shared = n_shared
        self.n_arm = n_arm
        self.alpha = alpha
        self.l2_lambda = l2_lambda
        self.gamma = gamma

        # Shared ridge: owned by the cluster, updated by all arms
        self._shared_ridge = shared_ridge

        # Per-arm ridge: owned by this arm
        self._arm_ridge: RidgeRegression | None = (
            RidgeRegression(n_features=n_arm, l2_lambda=l2_lambda, gamma=gamma)
            if n_arm > 0
            else None
        )

    def _split(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split context into (z_shared, x_arm)."""
        return x[: self.n_shared], x[self.n_shared :]

    def score(self, x: np.ndarray) -> float:
        """LinUCB-Hybrid score: expected reward + UCB width.

        Args:
            x: Full context vector, length n_shared + n_arm.
        Returns:
            UCB score (higher → arm preferred).
        """
        z, x_arm = self._split(x)

        expected = 0.0
        var = 0.0

        if self.n_shared > 0:
            expected += float(z @ self._shared_ridge.beta)
            a0_inv_z = self._shared_ridge.A_inv @ z
            var += float(z @ a0_inv_z)

        if self.n_arm > 0 and self._arm_ridge is not None:
            expected += float(x_arm @ self._arm_ridge.beta)
            a_inv_x = self._arm_ridge.A_inv @ x_arm
            var += float(x_arm @ a_inv_x)

        return expected + self.alpha * float(np.sqrt(max(var, 0.0)))

    def update(self, x: np.ndarray, reward: float, weight: float = 1.0) -> None:
        """Update shared and per-arm ridge models.

        Args:
            x: Full context vector, length n_shared + n_arm.
            reward: Observed reward.
            weight: IPS importance weight.
        """
        z, x_arm = self._split(x)

        if self.n_shared > 0:
            self._shared_ridge.update(z, reward, weight)

        if self.n_arm > 0 and self._arm_ridge is not None:
            self._arm_ridge.update(x_arm, reward, weight)

        self.is_fitted = True

    def update_batch(
        self, x_batch: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
    ) -> None:
        ws = np.ones(len(y), dtype=np.float64) if weights is None else weights
        z_batch = x_batch[:, : self.n_shared]
        x_arm_batch = x_batch[:, self.n_shared :]

        if self.n_shared > 0:
            self._shared_ridge.update_batch(z_batch, y, ws)

        if self.n_arm > 0 and self._arm_ridge is not None:
            self._arm_ridge.update_batch(x_arm_batch, y, ws)

        self.is_fitted = True

    def reset(self) -> None:
        """Reset per-arm ridge only. Shared ridge is reset by the cluster owner."""
        if self._arm_ridge is not None:
            self._arm_ridge.reset()
        self.is_fitted = False

    @property
    def beta_shared(self) -> np.ndarray:
        """Shared coefficient vector (cross-arm learned)."""
        return self._shared_ridge.beta

    @property
    def beta_arm(self) -> np.ndarray | None:
        """Per-arm coefficient vector."""
        return self._arm_ridge.beta if self._arm_ridge is not None else None
