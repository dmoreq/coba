"""

UCB1 arm model — context-free Upper Confidence Bound.

Reference:
  Auer et al., "Finite-time Analysis of the Multiarmed Bandit Problem", ML 2002.

Score formula:
  score_i = mu_i + alpha * sqrt(2 * ln(N) / n_i)

Where:
  - mu_i  is the empirical mean reward for arm i
  - N     is the total number of pulls across all arms
  - n_i   is the number of pulls for arm i
  - alpha controls exploration strength

UCB1 is context-free and serves as a strong baseline when the context model
is not yet reliable (cold start phase).
"""

import math

import numpy as np

from coba.policies.base import BaseArmModel
from coba.types import Arm

# A very large score assigned to arms that have never been pulled.
# This ensures all arms are explored at least once before exploitation begins.
_UNEXPLORED_SCORE = float("inf")


class UCB1ArmModel(BaseArmModel):
    """Per-arm UCB1 model with running mean and pull count.

    The `total_pulls` counter must be kept in sync across all arms by the
    parent bandit (passed via `set_total_pulls` or through the `score` call).

    Args:
        arm: Identifier for this arm.
        alpha: Exploration multiplier. Default 1.0 (standard UCB1).
        rng: NumPy random generator.
    """

    def __init__(
        self,
        arm: Arm,
        alpha: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(arm, rng or np.random.default_rng())
        self.alpha = alpha
        self._reward_sum: float = 0.0
        self._n_pulls: int = 0
        self._effective_pulls: float = 0.0

    def score(  # type: ignore[override]
        self, x: np.ndarray | None = None, total_pulls: int = 1
    ) -> float:
        """Compute UCB1 score.

        Args:
            x: Context vector (ignored — UCB1 is context-free).
            total_pulls: Total pulls across ALL arms (passed by the parent bandit).
        Returns:
            UCB1 score. Returns +inf for unexplored arms to force exploration.
        """
        if self._n_pulls == 0:
            return _UNEXPLORED_SCORE
        mu, exploration_bonus = self.score_decomposed(x, total_pulls=total_pulls)
        return mu + exploration_bonus

    def score_decomposed(
        self, x: np.ndarray | None = None, total_pulls: int = 1
    ) -> tuple[float, float]:
        """Return empirical mean and UCB1 exploration bonus separately.

        Args:
            x: Context vector (ignored).
            total_pulls: Total pulls across ALL arms.
        Returns:
            Tuple of (mu, exploration_bonus).
        """
        if self._n_pulls == 0:
            return 0.0, _UNEXPLORED_SCORE

        mu = self._reward_sum / max(self._effective_pulls, 1e-12)
        # Standard UCB1 confidence width
        exploration_bonus = self.alpha * math.sqrt(
            (2.0 * math.log(max(total_pulls, 1))) / max(self._effective_pulls, 1e-12)
        )
        return mu, exploration_bonus

    def update(self, x: np.ndarray | None, reward: float, weight: float = 1.0) -> None:
        """Update running statistics.

        Args:
            x: Context vector (ignored).
            reward: Observed scalar reward.
            weight: Importance weight. For UCB1 we scale the effective reward.
        """
        self._reward_sum += weight * reward
        self._n_pulls += 1
        self._effective_pulls += float(max(weight, 0.0))
        self.is_fitted = True

    def reset(self) -> None:
        """Re-initialize counters to zero."""
        self._reward_sum = 0.0
        self._n_pulls = 0
        self._effective_pulls = 0.0
        self.is_fitted = False

    @property
    def n_obs(self) -> int:
        return self._n_pulls

    @property
    def mean_reward(self) -> float:
        """Empirical mean reward."""
        if self._n_pulls == 0:
            return 0.0
        return self._reward_sum / max(self._effective_pulls, 1e-12)
