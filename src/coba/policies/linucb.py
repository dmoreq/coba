"""

LinUCB arm model — Linear Upper Confidence Bound.

Reference:
  Chu et al., "Contextual bandits with linear payoff functions", AISTATS 2011.
  Li et al., "A contextual-bandit approach to personalized news", WWW 2010.

Score formula:
  score(x) = x @ beta + alpha * sqrt(x @ A_inv @ x^T)

Where:
  - x @ beta  is the expected reward (exploitation term)
  - alpha * sqrt(x @ A_inv @ x^T)  is the upper confidence bound (exploration term)
  - alpha controls the exploration-exploitation trade-off

Higher alpha → more exploration. Recommended starting value: 0.5–2.0.
"""

import numpy as np

from coba.policies.base import BaseArmModel
from coba.policies.ridge import RidgeRegression
from coba.types import Arm


class LinUCBArmModel(BaseArmModel):
    """Per-arm LinUCB model backed by online ridge regression.

    Args:
        arm: Identifier for this price level arm.
        n_features: Context vector dimensionality.
        alpha: Exploration parameter (UCB width multiplier).
        l2_lambda: L2 regularization strength.
        gamma: Discount factor for non-stationary environments.
        rng: NumPy random generator (passed from parent bandit).
    """

    def __init__(
        self,
        arm: Arm,
        n_features: int,
        alpha: float = 1.0,
        l2_lambda: float = 1.0,
        gamma: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(arm, rng or np.random.default_rng())
        self.n_features = n_features
        self.alpha = alpha
        self.l2_lambda = l2_lambda
        self.gamma = gamma
        self._ridge = RidgeRegression(
            n_features=n_features, l2_lambda=l2_lambda, gamma=gamma
        )

    def score(self, x: np.ndarray) -> float:
        """Compute LinUCB score: E[y|x] + alpha * confidence_width(x).

        Args:
            x: Context vector, shape (n_features,).
        Returns:
            Upper confidence bound score (higher → arm is preferred).
        """
        # A_inv @ x, reused in both expected reward and UCB term
        a_inv_x = self._ridge.A_inv @ x  # shape (d,)

        # Expected reward from the learned linear model
        expected_reward = float(x @ self._ridge.beta)

        # UCB width: alpha * sqrt(x^T A_inv x)
        # The term x^T A_inv x is the "uncertainty" — larger for less-explored directions
        ucb_width = self.alpha * float(np.sqrt(max(x @ a_inv_x, 0.0)))

        return expected_reward + ucb_width

    def update(self, x: np.ndarray, reward: float, weight: float = 1.0) -> None:
        """Update the ridge regression model with a new (context, reward) pair.

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
        """Batch update the ridge regression model."""
        self._ridge.update_batch(x_batch, y, weights)
        self.is_fitted = True

    def reset(self) -> None:
        """Re-initialize to prior (λI identity, zero Xty)."""
        self._ridge.reset()
        self.is_fitted = False

    @property
    def n_obs(self) -> int:
        return self._ridge.n_obs

    @property
    def beta(self) -> np.ndarray:
        """Learned coefficient vector (for inspection/debugging)."""
        return self._ridge.beta
