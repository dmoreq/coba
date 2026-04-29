"""

Ridge Regression with online Sherman-Morrison (rank-1) updates.

This is the shared statistical backbone for LinUCB and LinTS.

Key design decision (from contextualbandits analysis):
  The Sherman-Morrison formula lets us update A_inv incrementally in O(d²)
  per observation, avoiding the O(d³) cost of np.linalg.inv on every update.
  This is critical for real-time online learning in a streaming environment.

  Update rule for a new observation (x, y):
    A  ← A + w * x x^T
    A_inv ← A_inv - w * (A_inv x x^T A_inv) / (1 + w * x^T A_inv x)
    Xty ← Xty + w * x * y
    beta ← A_inv @ Xty
"""

import numpy as np


class RidgeRegression:
    """Online ridge regression with Sherman-Morrison incremental updates.

    Maintains sufficient statistics (A, A_inv, Xty) that allow O(d²) updates
    per observation without recomputing the matrix inverse from scratch.

    Args:
        n_features: Dimensionality of the context vector.
        l2_lambda: L2 regularization strength (λ). Higher → more shrinkage.
    """

    def __init__(self, n_features: int, l2_lambda: float = 1.0, gamma: float = 1.0) -> None:
        self.n_features = n_features
        self.l2_lambda = l2_lambda
        self.gamma = gamma

        # A_inv = (λI)^-1 initially = (1/λ) I
        self.A_inv: np.ndarray = (1.0 / l2_lambda) * np.eye(n_features, dtype=np.float64)

        # Xty = Σ(w * x_i * y_i)   [response accumulator]
        self.Xty: np.ndarray = np.zeros(n_features, dtype=np.float64)

        # beta = A_inv @ Xty   [coefficient vector]
        self.beta: np.ndarray = np.zeros(n_features, dtype=np.float64)

        self.n_obs: int = 0

    @property
    def a_matrix(self) -> np.ndarray:
        """Design matrix A = A_inv^{-1} (for debugging/testing only).

        Warning:
            This is an O(d³) operation. Never call this on the hot path.
            Exposed only so tests can assert the initial state matches λI.
        """
        return np.linalg.inv(self.A_inv)

    # Convenience alias kept for backwards-compat with existing tests
    A = a_matrix

    def update(self, x: np.ndarray, y: float, weight: float = 1.0) -> None:
        """Incorporate one observation using Sherman-Morrison rank-1 update.

        Args:
            x: Context vector, shape (n_features,).
            y: Scalar reward observation.
            weight: Importance weight (1/propensity for IPS). Default 1.0.
        """
        # Apply discount factor for non-stationary environments
        if self.gamma < 1.0:
            self.A_inv /= self.gamma
            self.Xty *= self.gamma

        # Compute A_inv @ x once to reuse in both numerator and denominator
        a_inv_x = self.A_inv @ x  # shape (d,)

        # Sherman-Morrison denominator: scalar
        denom = 1.0 + weight * float(x @ a_inv_x)

        # Rank-1 downdate of A_inv: A_inv ← A_inv - w*(A_inv x x^T A_inv) / denom
        self.A_inv -= weight * np.outer(a_inv_x, a_inv_x) / denom

        # Update response accumulator
        self.Xty += weight * x * y

        # Recompute coefficients
        self.beta = self.A_inv @ self.Xty

        self.n_obs += 1

    def _update_no_beta(self, x: np.ndarray, y: float, weight: float = 1.0) -> None:
        """SM update without recomputing beta (deferred to batch end)."""
        if self.gamma < 1.0:
            self.A_inv /= self.gamma
            self.Xty *= self.gamma
        a_inv_x = self.A_inv @ x
        denom = 1.0 + weight * float(x @ a_inv_x)
        self.A_inv -= weight * np.outer(a_inv_x, a_inv_x) / denom
        self.Xty += weight * x * y
        self.n_obs += 1

    def update_batch(
        self, x_batch: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
    ) -> None:
        """Batch update by iterating Sherman-Morrison over all observations.

        For large batches a Cholesky recompute would be faster, but SM is simpler
        and sufficient for the typical batch sizes seen in online learning (< 10k rows).

        Args:
            x_batch: Context matrix, shape (n_samples, n_features).
            y: Reward vector, shape (n_samples,).
            weights: Importance weights, shape (n_samples,). None → all ones.
        """
        ws = np.ones(len(y), dtype=np.float64) if weights is None else weights
        for xi, yi, wi in zip(x_batch, y, ws):
            self._update_no_beta(xi, float(yi), float(wi))
        self.beta = self.A_inv @ self.Xty

    def predict(self, x: np.ndarray) -> float:
        """Compute expected reward: E[y | x] = x @ beta.

        Args:
            x: Context vector, shape (n_features,).
        Returns:
            Scalar expected reward.
        """
        return float(x @ self.beta)

    def reset(self, l2_lambda: float | None = None) -> None:
        """Re-initialize to prior state (all-zero Xty, λI matrix)."""
        lam = l2_lambda if l2_lambda is not None else self.l2_lambda
        self.A_inv = (1.0 / lam) * np.eye(self.n_features, dtype=np.float64)
        self.Xty = np.zeros(self.n_features, dtype=np.float64)
        self.beta = np.zeros(self.n_features, dtype=np.float64)
        self.n_obs = 0
