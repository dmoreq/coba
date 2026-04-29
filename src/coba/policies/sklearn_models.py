"""
Non-linear Arm Models wrapping Scikit-Learn compatible estimators.

These models allow the use of arbitrary regressors (like SGDRegressor,
LGBMRegressor) by applying meta-heuristics like Epsilon-Greedy,
Bootstrapped Thompson Sampling, and Bootstrapped UCB.
"""

import copy
from typing import Any

import numpy as np

from coba.policies.base import BaseArmModel
from coba.types import Arm

# Scores returned to force exploration for unfitted arms or epsilon-greedy trials.
# Set deliberately high so any fitted arm with a real prediction will be preferred
# once the model has enough data to trust its estimates.
_EXPLORE_SCORE_LOW: float = 1e3
_EXPLORE_SCORE_HIGH: float = 1e4


class EpsilonGreedyArmModel(BaseArmModel):
    """Epsilon-Greedy policy wrapping a single base estimator.

    With probability 1 - epsilon, it predicts the expected reward using the base estimator.
    With probability epsilon, it explores by returning a very high random score to force selection.

    Args:
        arm: Identifier for the arm.
        rng: Random number generator.
        base_estimator: A Scikit-Learn compatible estimator implementing `partial_fit` and `predict`.
        epsilon: Probability of exploration (0.0 to 1.0).
    """

    def __init__(
        self,
        arm: Arm,
        rng: np.random.Generator,
        base_estimator: Any,
        epsilon: float = 0.1,
    ) -> None:
        super().__init__(arm, rng)
        self.epsilon = epsilon

        # We need a fresh copy of the base estimator so it is uninitialized
        if base_estimator is None:
            raise ValueError("base_estimator must be provided for EpsilonGreedyArmModel")

        self.model = copy.deepcopy(base_estimator)
        self.is_fitted = False

    def score(self, x: np.ndarray) -> float:
        """Returns the predicted reward, or a large value for exploration."""
        if not self.is_fitted:
            # Optimistic initialization for exploration
            return self.rng.uniform(_EXPLORE_SCORE_LOW, _EXPLORE_SCORE_HIGH)

        if self.rng.random() < self.epsilon:
            # Force exploration: return a large score so this arm gets picked
            return self.rng.uniform(_EXPLORE_SCORE_LOW, _EXPLORE_SCORE_HIGH)

        # Exploitation: predict using the underlying model
        pred = self.model.predict(x.reshape(1, -1))[0]
        return float(pred)

    def update(self, x: np.ndarray, reward: float, weight: float = 1.0) -> None:
        """Update the model using a single sample with the given weight."""
        self.model.partial_fit(x.reshape(1, -1), [reward], sample_weight=[weight])
        self.is_fitted = True

    def update_batch(
        self, x_batch: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
    ) -> None:
        """Batch update the model."""
        ws = np.ones(len(y)) if weights is None else weights
        for xi, yi, wi in zip(x_batch, y, ws):
            self.update(xi, float(yi), float(wi))

    def reset(self) -> None:
        """Reset the model by re-instantiating the estimator."""
        from sklearn.base import clone

        self.model = clone(self.model)
        self.is_fitted = False


class _BootstrappedArmModelBase(BaseArmModel):
    """Base class for Bootstrapped arm models (TS and UCB).

    Maintains an ensemble of N estimators and updates them online using
    Poisson or Gamma distributed weights.
    """

    def __init__(
        self,
        arm: Arm,
        rng: np.random.Generator,
        base_estimator: Any,
        n_bootstraps: int = 10,
        bootstrap_method: str = "poisson",
    ) -> None:
        super().__init__(arm, rng)
        self.n_bootstraps = n_bootstraps
        self.bootstrap_method = bootstrap_method

        if base_estimator is None:
            raise ValueError("base_estimator must be provided for bootstrapped models")

        self.models = [copy.deepcopy(base_estimator) for _ in range(self.n_bootstraps)]
        self.is_fitted = False

    def update(self, x: np.ndarray, reward: float, weight: float = 1.0) -> None:
        """Update each estimator in the ensemble with a bootstrapped weight."""
        x_2d = x.reshape(1, -1)
        y_1d = [reward]

        for model in self.models:
            # Draw bootstrap weight
            if self.bootstrap_method == "poisson":
                w = self.rng.poisson(1.0) * weight
            else:
                w = self.rng.gamma(1.0, 1.0) * weight

            if w > 0:
                model.partial_fit(x_2d, y_1d, sample_weight=[w])

        self.is_fitted = True

    def update_batch(
        self, x_batch: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
    ) -> None:
        """Batch update the ensemble with bootstrapped weights for each sample."""
        n_samples = len(y)
        if weights is None:
            weights = np.ones(n_samples)

        for model in self.models:
            if self.bootstrap_method == "poisson":
                boot_w = self.rng.poisson(1.0, size=n_samples) * weights
            else:
                boot_w = self.rng.gamma(1.0, 1.0, size=n_samples) * weights

            # Filter out zero weights for efficiency if needed, but partial_fit handles it
            model.partial_fit(x_batch, y, sample_weight=boot_w)

        self.is_fitted = True

    def reset(self) -> None:
        """Reset all estimators in the ensemble."""
        from sklearn.base import clone

        self.models = [clone(m) for m in self.models]
        self.is_fitted = False

    def _predict_all(self, x: np.ndarray) -> np.ndarray:
        """Return predictions from all models in the ensemble."""
        if not self.is_fitted:
            # Optimistic initialization — encourages equal exploration of all arms
            return self.rng.uniform(_EXPLORE_SCORE_LOW, _EXPLORE_SCORE_HIGH, size=self.n_bootstraps)

        x_2d = x.reshape(1, -1)
        preds = [model.predict(x_2d)[0] for model in self.models]
        return np.array(preds)


class BootstrappedTSArmModel(_BootstrappedArmModelBase):
    """Bootstrapped Thompson Sampling Arm Model.

    Predicts by sampling one random model's prediction from the ensemble.
    """

    def score(self, x: np.ndarray) -> float:
        """Sample one prediction from the bootstrapped ensemble."""
        preds = self._predict_all(x)
        return float(self.rng.choice(preds))


class BootstrappedUCBArmModel(_BootstrappedArmModelBase):
    """Bootstrapped Upper Confidence Bound Arm Model.

    Predicts by returning the p-th percentile of predictions from the ensemble.
    """

    def __init__(
        self,
        arm: Arm,
        rng: np.random.Generator,
        base_estimator: Any,
        n_bootstraps: int = 10,
        bootstrap_method: str = "poisson",
        percentile: float = 80.0,
    ) -> None:
        super().__init__(arm, rng, base_estimator, n_bootstraps, bootstrap_method)
        self.percentile = percentile

    def score(self, x: np.ndarray) -> float:
        """Return the p-th percentile of the ensemble's predictions."""
        preds = self._predict_all(x)
        return float(np.percentile(preds, self.percentile))
