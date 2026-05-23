"""
GP-UCB arm model — Gaussian Process Upper Confidence Bound.

Reference:
  Srinivas et al., "Gaussian Process Optimization in the Bandit Setting:
  No Regret and Experimental Design", ICML 2010.

Score formula:
  score(x) = μ(x) + beta * σ(x)

Where:
  μ(x) = k(x, X_obs) @ (K + noise_var * I)^{-1} @ y_obs   (posterior mean)
  σ(x) = sqrt(k(x,x) - k(x, X_obs) @ (K + noise_var * I)^{-1} @ k(X_obs, x))

Kernel: RBF (Radial Basis Function) k(x, x') = exp(-||x - x'||² / (2 * length_scale²))

Complexity:
  - Fit:    O(n³) — Cholesky decomposition of the n×n kernel matrix.
  - Score:  O(n²) — kernel vector + matrix-vector product.
  - Memory: O(n²) — stores the full kernel matrix inverse.

For this reason, GP-UCB is best for low-volume decisions where observations
are expensive (e.g. A/B test variants, hyperparameter search, drug discovery).
Use LinUCB or LinTS for high-throughput real-time serving.

The model subsamples to at most `max_obs` recent observations to keep inference
tractable in longer runs.
"""

from __future__ import annotations

import numpy as np

from coba.policies.base import BaseArmModel
from coba.types import Arm

_DEFAULT_MAX_OBS = 500


class GPUCBArmModel(BaseArmModel):
    """Per-arm GP-UCB model using RBF kernel.

    Args:
        arm: Arm identifier.
        beta: UCB exploration coefficient (width multiplier on posterior std).
              Higher → more exploration. Default 2.0.
        length_scale: RBF kernel bandwidth. Controls how quickly reward
                      correlation decays with distance. Default 1.0.
        noise_var: Observation noise variance (σ²). Equivalent to L2
                   regularisation in ridge regression. Default 0.1.
        max_obs: Maximum number of observations stored. Oldest observations
                 are dropped when exceeded to keep O(n²) tractable. Default 500.
        rng: NumPy random generator.
    """

    def __init__(
        self,
        arm: Arm,
        beta: float = 2.0,
        length_scale: float = 1.0,
        noise_var: float = 0.1,
        max_obs: int = _DEFAULT_MAX_OBS,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(arm, rng or np.random.default_rng())
        self.beta = beta
        self.length_scale = length_scale
        self.noise_var = noise_var
        self.max_obs = max_obs

        self._X: list[np.ndarray] = []  # observed contexts
        self._y: list[float] = []  # observed rewards
        self._w: list[float] = []  # importance weights

        # Cached Cholesky decomposition: L such that L @ L.T = K + noise_var * I
        self._L: np.ndarray | None = None
        self._alpha_vec: np.ndarray | None = None  # (K + σ²I)^{-1} @ y
        self._dirty: bool = True  # True when cache needs rebuild

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, x: np.ndarray) -> float:
        """GP-UCB score: μ(x) + beta * σ(x).

        Returns float("inf") before any observations (cold start) to guarantee
        each arm is explored at least once.
        """
        if not self._X:
            return float("inf")

        mu, ucb_width = self.score_decomposed(x)
        return mu + ucb_width

    def score_decomposed(self, x: np.ndarray) -> tuple[float, float]:
        """Return posterior mean and exploration width separately.

        Args:
            x: Context vector.
        Returns:
            Tuple of (mu, ucb_width).
        """
        if not self._X:
            return 0.0, float("inf")

        self._maybe_rebuild_cache()

        x_obs = np.array(self._X)  # (n, d)
        k_vec = self._rbf_kernel_vec(x, x_obs)  # (n,)

        # Posterior mean: k_vec @ (K + σ²I)^{-1} @ y
        mu = float(k_vec @ self._alpha_vec)

        # Posterior variance: k(x,x) - k_vec @ (K + σ²I)^{-1} @ k_vec
        # Use Cholesky: v = L^{-1} k_vec → var = k(x,x) - ||v||²
        v = np.linalg.solve(self._L, k_vec)
        k_xx = 1.0  # RBF self-kernel is always 1
        var = max(k_xx - float(v @ v), 0.0)

        return mu, self.beta * float(np.sqrt(var))

    def update(self, x: np.ndarray, reward: float, weight: float = 1.0) -> None:
        """Add a new (context, reward) observation.

        Marks cache dirty — Cholesky is rebuilt lazily at next score() call.
        """
        self._X.append(x.copy())
        self._y.append(reward * weight)  # IPS-weighted reward
        self._w.append(weight)

        # Trim to max_obs (FIFO)
        if len(self._X) > self.max_obs:
            self._X = self._X[-self.max_obs :]
            self._y = self._y[-self.max_obs :]
            self._w = self._w[-self.max_obs :]

        self._dirty = True
        self.is_fitted = True

    def reset(self) -> None:
        """Clear all observations and cached kernel matrix."""
        self._X = []
        self._y = []
        self._w = []
        self._L = None
        self._alpha_vec = None
        self._dirty = True
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _rbf_kernel_vec(self, x: np.ndarray, x_obs: np.ndarray) -> np.ndarray:
        """Compute RBF kernel between x and all rows of x_obs.

        Args:
            x: Query point, shape (d,).
            x_obs: Observed points, shape (n, d).
        Returns:
            Kernel vector, shape (n,).
        """
        diff = x_obs - x  # (n, d)
        sq_dist = np.sum(diff**2, axis=1)  # (n,)
        return np.exp(-sq_dist / (2.0 * self.length_scale**2))

    def _rbf_kernel_matrix(self, x_obs: np.ndarray) -> np.ndarray:
        """Compute RBF kernel matrix for x_obs.

        Args:
            x_obs: Observed points, shape (n, d).
        Returns:
            Kernel matrix, shape (n, n).
        """
        # ||xi - xj||² = ||xi||² + ||xj||² - 2 xi·xj  (vectorised)
        sq_norms = np.sum(x_obs**2, axis=1, keepdims=True)  # (n, 1)
        sq_dists = sq_norms + sq_norms.T - 2.0 * (x_obs @ x_obs.T)
        sq_dists = np.maximum(sq_dists, 0.0)  # guard fp noise
        return np.exp(-sq_dists / (2.0 * self.length_scale**2))

    def _maybe_rebuild_cache(self) -> None:
        """Rebuild Cholesky cache if observations changed since last call."""
        if not self._dirty or not self._X:
            return

        x_obs = np.array(self._X)  # (n, d)
        y_obs = np.array(self._y)  # (n,)

        k_matrix = self._rbf_kernel_matrix(x_obs)  # (n, n)
        noisy_k = k_matrix + self.noise_var * np.eye(len(x_obs))

        # Cholesky: L L.T = K + σ²I  (more numerically stable than direct inv)
        try:
            self._L = np.linalg.cholesky(noisy_k)
        except np.linalg.LinAlgError:
            # Add jitter if Cholesky fails due to near-singular matrix
            noisy_k += 1e-6 * np.eye(len(x_obs))
            self._L = np.linalg.cholesky(noisy_k)

        # alpha = (K + σ²I)^{-1} y  via two triangular solves
        self._alpha_vec = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y_obs))
        self._dirty = False
