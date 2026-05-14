"""
Softmax (Boltzmann) Arm Model.

Instead of the hard-argmax selection used by UCB/TS variants, Softmax converts
scores into selection probabilities proportional to exp(score / tau):

    p(a | x) = exp(score_a(x) / tau) / Σ_b exp(score_b(x) / tau)

This is especially useful in recommendation systems where you want stochastic
output proportional to quality (e.g. for diversity, fairness, or A/B testing),
rather than always returning the single top-scoring option.

The ``score()`` method returns the Softmax-transformed probability so that
ClusterRouter's argmax selection is equivalent to sampling from the softmax
distribution when used inside ClusterBandit.decide().

For true stochastic sampling (rather than argmax on the softmax probabilities),
use ClusterBandit.decide() with a wrapper that samples proportionally — this is
left to the caller so the core API remains deterministic and testable.

Temperature (tau) controls exploration:
  - tau → 0  : converges to greedy (argmax) selection
  - tau → ∞  : converges to uniform random selection
  - tau = 1.0: default; reasonable starting point

The underlying reward estimator is a per-arm ridge regression (same as LinUCB)
so the model learns contextual reward expectations.
"""

from __future__ import annotations

import numpy as np

from coba.policies.base import _RidgeBackedArmModel
from coba.types import Arm

_COLD_START_SCORE: float = float("inf")


class SoftmaxArmModel(_RidgeBackedArmModel):
    """Per-arm Softmax model — returns exp(expected_reward / tau) as the score.

    Because ClusterRouter calls score() independently per arm and then takes
    the argmax, the resulting selection is equivalent to argmax of softmax
    probabilities (which is still the greedy argmax). For genuine stochastic
    sampling, the caller should use the ``ClusterBandit.score_all()`` output
    and sample proportionally.

    Args:
        arm: Arm identifier.
        n_features: Context vector dimensionality.
        tau: Temperature parameter. Lower → more greedy. Default 1.0.
        l2_lambda: L2 regularization strength.
        gamma: Discount factor for non-stationarity.
        rng: NumPy random generator.
    """

    def __init__(
        self,
        arm: Arm,
        n_features: int,
        tau: float = 1.0,
        l2_lambda: float = 1.0,
        gamma: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(arm, n_features, l2_lambda, gamma, rng)
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        self.tau = tau

    def score(self, x: np.ndarray) -> float:
        """Return exp(E[reward | x] / tau).

        Cold start: return +inf so the arm is always explored at least once.
        """
        if not self.is_fitted:
            return _COLD_START_SCORE
        expected_reward = float(x @ self._ridge.beta)
        # Numerically stable: clamp the exponent to avoid overflow
        return float(np.exp(np.clip(expected_reward / self.tau, -50.0, 50.0)))
