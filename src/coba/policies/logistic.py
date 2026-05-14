"""
Logistic Bandits with Online Laplace Approximation.

Provides Thompson Sampling (TS) and Upper Confidence Bound (UCB) variants
for binary reward environments (e.g., Conversion Rate, Click-Through Rate).

Uses an online 1-step Newton-Raphson update with Sherman-Morrison to
maintain O(d^2) complexity, suitable for high-throughput real-time systems.
"""

import numpy as np

from coba.policies.base import _LogisticBackedArmModel
from coba.types import Arm


def sigmoid(x: float) -> float:
    """Stable sigmoid implementation."""
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = np.exp(x)
        return z / (1.0 + z)


class OnlineLogisticRegression:
    """Online Logistic Regression using Laplace Approximation.

    Maintains the parameter vector `w` and the inverse Hessian `H_inv`
    (which acts as the covariance matrix of the Gaussian posterior).
    """

    def __init__(self, n_features: int, l2_lambda: float = 1.0, gamma: float = 1.0) -> None:
        self.n_features = n_features
        self.l2_lambda = l2_lambda
        self.gamma = gamma

        # Inverse Hessian (covariance matrix) initialized with L2 prior
        self.H_inv: np.ndarray = (1.0 / l2_lambda) * np.eye(n_features, dtype=np.float64)

        # Weight vector
        self.w: np.ndarray = np.zeros(n_features, dtype=np.float64)
        self.n_obs: int = 0

    def update(self, x: np.ndarray, y: float, weight: float = 1.0) -> None:
        """Update using 1-step online Newton-Raphson.

        Args:
            x: Context vector.
            y: Binary reward (0.0 or 1.0).
            weight: Importance weight.
        """
        # Apply discount factor for non-stationary environments
        if self.gamma < 1.0:
            self.H_inv /= self.gamma
            # Note: w is not directly scaled like Xty in Ridge because
            # w represents the actual parameters, not an accumulator.

        # Predict probability using current weights
        pred = sigmoid(float(x @ self.w))

        # Variance of the Bernoulli distribution for this prediction
        v = pred * (1.0 - pred) * weight

        # Prevent v from being exactly 0 to avoid degenerate Hessian updates
        v = max(v, 1e-6)

        # Sherman-Morrison rank-1 downdate on the inverse Hessian.
        # Clamp denominator — same drift guard as ridge.py.
        h_inv_x = self.H_inv @ x
        denom = max(1.0 + v * float(x @ h_inv_x), 1e-10)
        self.H_inv -= v * np.outer(h_inv_x, h_inv_x) / denom

        # Gradient of the log-likelihood for this sample
        g = weight * (y - pred) * x

        # 1-step Newton update: w = w + H_inv @ g
        self.w += self.H_inv @ g
        self.n_obs += 1

    def update_batch(
        self, x_batch: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
    ) -> None:
        """Batch update by iterating the online update."""
        ws = np.ones(len(y)) if weights is None else weights
        for xi, yi, wi in zip(x_batch, y, ws):
            self.update(xi, float(yi), float(wi))

    def predict_proba(self, x: np.ndarray) -> float:
        """Return the expected probability."""
        return sigmoid(float(x @ self.w))

    def reset(self, l2_lambda: float | None = None) -> None:
        """Reset the model to its prior state."""
        lam = l2_lambda if l2_lambda is not None else self.l2_lambda
        self.H_inv = (1.0 / lam) * np.eye(self.n_features, dtype=np.float64)
        self.w = np.zeros(self.n_features, dtype=np.float64)
        self.n_obs = 0


class LogisticUCBArmModel(_LogisticBackedArmModel):
    """Logistic UCB using Laplace Approximation.

    Computes the UCB in the logit space and applies the sigmoid function.
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
        """Compute the Upper Confidence Bound for the probability."""
        logit_mu = float(x @ self.model.w)
        variance = float(x @ self.model.H_inv @ x)
        ucb_width = self.alpha * np.sqrt(max(variance, 0.0))
        return sigmoid(logit_mu + ucb_width)


class LogisticTSArmModel(_LogisticBackedArmModel):
    """Logistic Thompson Sampling using Laplace Approximation.

    Samples weights from the Gaussian posterior and returns the probability.
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
        super().__init__(arm, n_features, l2_lambda, gamma, rng)
        self.v_sq = v_sq

    def score(self, x: np.ndarray) -> float:
        """Sample coefficients from posterior and return predicted probability."""
        cov = self.v_sq * self.model.H_inv
        cov = (cov + cov.T) / 2.0
        min_diag = 1e-10
        cov[np.diag_indices_from(cov)] = np.maximum(np.diag(cov), min_diag)
        chol_l = np.linalg.cholesky(cov)
        z = self.rng.standard_normal(self.n_features)
        w_sample = self.model.w + chol_l @ z
        return sigmoid(float(x @ w_sample))
