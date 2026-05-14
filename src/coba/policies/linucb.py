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

from coba.policies.base import _RidgeBackedArmModel
from coba.types import Arm


class LinUCBArmModel(_RidgeBackedArmModel):
    """Per-arm LinUCB model backed by online ridge regression.

    Args:
        arm: Identifier for this arm.
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
        super().__init__(arm, n_features, l2_lambda, gamma, rng)
        self.alpha = alpha

    def score(self, x: np.ndarray) -> float:
        """Compute LinUCB score: E[y|x] + alpha * confidence_width(x).

        Args:
            x: Context vector, shape (n_features,).
        Returns:
            Upper confidence bound score (higher → arm is preferred).
        """
        mean_est, ucb_width = self.score_decomposed(x)
        return mean_est + ucb_width

    def score_decomposed(self, x: np.ndarray) -> tuple[float, float]:
        """Return (mean_estimate, confidence_width) separately.

        Useful for monitoring and for populating BanditDecision.mean_estimate /
        BanditDecision.confidence_width without re-computing the UCB.

        Args:
            x: Context vector, shape (n_features,).
        Returns:
            Tuple of (expected_reward, ucb_width).
        """
        a_inv_x = self._ridge.A_inv @ x
        mean_est = float(x @ self._ridge.beta)
        ucb_width = self.alpha * float(np.sqrt(max(x @ a_inv_x, 0.0)))
        return mean_est, ucb_width
