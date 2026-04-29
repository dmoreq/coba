"""

LinTS arm model — Linear Thompson Sampling.

Reference:
  Agrawal & Goyal, "Thompson Sampling for Contextual Bandits with Linear Payoffs",
  ICML 2013.

Sampling strategy:
  beta_sample ~ MVN(beta_hat, v_sq * A_inv)
  score(x) = x @ beta_sample

Where v_sq controls the variance of the prior — higher v_sq → more exploration.
This naturally produces calibrated exploration without a fixed UCB parameter,
making it preferable in non-stationary environments.

Implementation note:
  We use np.random.Generator.multivariate_normal which is faster than
  scipy's implementation and uses the Cholesky decomposition internally.
  For stability we clip the diagonal of A_inv * v_sq to avoid negative values
  from floating point errors before Cholesky decomposition.
"""

import numpy as np

from coba.policies.base import BaseArmModel
from coba.policies.ridge import RidgeRegression
from coba.types import Arm


class LinTSArmModel(BaseArmModel):
    """Per-arm LinTS model — Thompson Sampling via multivariate normal sampling.

    Args:
        arm: Identifier for this arm.
        n_features: Context vector dimensionality.
        v_sq: Variance multiplier for the posterior covariance. Higher → more exploration.
        l2_lambda: L2 regularization strength.
        gamma: Discount factor for non-stationary environments.
        rng: NumPy random generator (seeded from parent bandit).
    """

    def __init__(
        self,
        arm: Arm,
        n_features: int,
        v_sq: float = 1.0,
        l2_lambda: float = 1.0,
        gamma: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(arm, rng or np.random.default_rng())
        self.n_features = n_features
        self.v_sq = v_sq
        self.l2_lambda = l2_lambda
        self.gamma = gamma
        self._ridge = RidgeRegression(n_features=n_features, l2_lambda=l2_lambda, gamma=gamma)

    def score(self, x: np.ndarray) -> float:
        """Sample coefficients from posterior and return dot product with context.

        This is the Thompson Sampling step: each call samples a different set of
        coefficients, which produces natural exploration proportional to uncertainty.

        Args:
            x: Context vector, shape (n_features,).
        Returns:
            Sampled expected reward (scalar).
        """
        cov = self.v_sq * self._ridge.A_inv

        # Ensure positive semi-definiteness by symmetrizing and clipping diagonal.
        # Floating point accumulation in SM updates can introduce tiny negative values.
        cov = (cov + cov.T) / 2.0
        min_diag = 1e-10
        cov[np.diag_indices_from(cov)] = np.maximum(np.diag(cov), min_diag)

        # Sample using Cholesky decomposition: x @ (mu + L @ z) = x @ mu + (x @ L) @ z
        chol_l = np.linalg.cholesky(cov)
        z = self.rng.standard_normal(self.n_features)
        return float(x @ self._ridge.beta + (x @ chol_l) @ z)

    def update(self, x: np.ndarray, reward: float, weight: float = 1.0) -> None:
        """Update the ridge model with a new (context, reward) pair.

        Args:
            x: Context vector, shape (n_features,).
            reward: Observed scalar reward.
            weight: IPS importance weight (1/propensity). Default 1.0.
        """
        self._ridge.update(x, reward, weight)
        self.is_fitted = True

    def update_batch(
        self, x_batch: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
    ) -> None:
        """Batch update the ridge model."""
        self._ridge.update_batch(x_batch, y, weights)
        self.is_fitted = True

    def reset(self) -> None:
        """Re-initialize to prior."""
        self._ridge.reset()
        self.is_fitted = False

    @property
    def n_obs(self) -> int:
        return self._ridge.n_obs

    @property
    def beta(self) -> np.ndarray:
        """MAP estimate of coefficient vector (for inspection/debugging)."""
        return self._ridge.beta
