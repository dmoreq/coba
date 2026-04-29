"""

Thompson Sampling arm model — context-free Beta posterior.

Appropriate as a fallback when there is no context, or when the number of
observations is too small to fit a reliable linear model.

The Beta(alpha, beta) posterior is maintained per arm:
  - alpha (success count): incremented by normalized reward
  - beta  (failure count): incremented by (1 - normalized reward)

Scores are sampled each call, producing natural exploration calibrated to
the uncertainty in the Beta posterior.

Note: rewards must be in [0, 1] (e.g. conversion rate, normalized engagement score).
"""

import numpy as np

from coba.policies.base import BaseArmModel
from coba.types import Arm


class ThompsonArmModel(BaseArmModel):
    """Per-arm context-free Thompson Sampling model with Beta posterior.

    Args:
        arm: Identifier for this arm.
        alpha_prior: Beta distribution alpha prior (pseudo-successes). Default 1.0.
        beta_prior: Beta distribution beta prior (pseudo-failures). Default 1.0.
        rng: NumPy random generator.
    """

    def __init__(
        self,
        arm: Arm,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(arm, rng or np.random.default_rng())
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

        # Running counts for the Beta posterior
        self._alpha: float = alpha_prior  # successes
        self._beta: float = beta_prior  # failures
        self._n_obs: int = 0

    def score(self, x: np.ndarray | None = None) -> float:  # type: ignore[override]
        """Sample from the Beta posterior.

        The context vector x is ignored (context-free policy), but the signature
        is kept consistent with BaseArmModel for interoperability.

        Returns:
            A sample from Beta(alpha, beta) in [0, 1].
        """
        return float(self.rng.beta(self._alpha, self._beta))

    def update(self, x: np.ndarray | None, reward: float, weight: float = 1.0) -> None:
        """Update the Beta posterior with the observed reward.

        The weight (IPS) is applied to the reward and (1-reward) increments,
        effectively scaling the effective number of observations.

        Args:
            x: Context vector (ignored for context-free Thompson Sampling).
            reward: Observed reward in [0, 1].
            weight: Importance weight. Default 1.0.
        """
        # Clip reward to [0, 1] to ensure valid Beta update
        r = float(np.clip(reward, 0.0, 1.0))
        self._alpha += weight * r
        self._beta += weight * (1.0 - r)
        self._n_obs += 1
        self.is_fitted = True

    def reset(self) -> None:
        """Re-initialize to the specified prior."""
        self._alpha = self.alpha_prior
        self._beta = self.beta_prior
        self._n_obs = 0
        self.is_fitted = False

    @property
    def n_obs(self) -> int:
        return self._n_obs

    @property
    def mean_reward(self) -> float:
        """Posterior mean: alpha / (alpha + beta)."""
        return self._alpha / (self._alpha + self._beta)
