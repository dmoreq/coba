"""
Abstract base class for all bandit arm models in coba.

Each arm maintains its own statistical model (posterior, regression, counters).
This is the "one model per arm" paradigm, consistent with the OvR approach in
contextualbandits and the arm_to_model dict in mabwiser.
"""

import abc
from typing import Any

import numpy as np

from coba.types import Arm


class BaseArmModel(abc.ABC):
    """Abstract per-arm model.

    Every arm in the bandit has one instance of a BaseArmModel subclass.
    The model is responsible for:
      - Maintaining its own sufficient statistics (A, Xty, beta, counts, etc.)
      - Providing a `score(x)` method used by the bandit to rank arms.
      - Updating itself given new (context, reward) observations via `update()`.
    """

    # Shared dynamic attributes exposed by concrete subclasses for observability.
    arm: Arm
    rng: np.random.Generator
    is_fitted: bool
    alpha: float
    beta: Any
    leaf: Any
    last_was_random: bool
    mean_reward: float
    n_obs: int
    score_decomposed: Any
    _X: Any
    _gamma: float
    _ridge: Any

    def __init__(self, arm: Arm, rng: np.random.Generator) -> None:
        self.arm = arm
        self.rng = rng
        self.is_fitted = False

    @abc.abstractmethod
    def score(self, x: np.ndarray) -> float:
        """Return the arm's score for context vector x.

        The score represents the "optimistic reward estimate" for this arm.
        Higher score → arm is preferred. Shapes: x is (n_features,).
        """

    @abc.abstractmethod
    def update(self, x: np.ndarray, reward: float, weight: float = 1.0) -> None:
        """Incorporate a new (context, reward) observation into the arm's model.

        Args:
            x: Context feature vector, shape (n_features,).
            reward: Observed scalar reward.
            weight: Importance weight (e.g. 1/propensity for IPS correction).
                    Default 1.0 means no correction.
        """

    def update_batch(
        self, x_batch: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
    ) -> None:
        """Batch update. Default implementation iterates over observations.

        Args:
            x_batch: Context matrix, shape (n_samples, n_features).
            y: Observed rewards, shape (n_samples,).
            weights: Importance weights.
        """
        ws = np.ones(len(y), dtype=np.float64) if weights is None else weights
        for xi, yi, wi in zip(x_batch, y, ws):
            self.update(xi, float(yi), float(wi))

    @abc.abstractmethod
    def reset(self) -> None:
        """Re-initialize the arm's model to its prior state (used in full refit)."""

    def clone(self) -> "BaseArmModel":
        """Return a deep copy of this arm model (used for warm-starting cold arms)."""
        import copy

        return copy.deepcopy(self)


class _RidgeBackedArmModel(BaseArmModel):
    """Shared boilerplate for arm models backed by RidgeRegression.

    Subclasses only need to implement ``score()``. All update / reset / property
    logic is provided here so it is never duplicated across LinUCB / LinTS.
    """

    def __init__(
        self,
        arm: Arm,
        n_features: int,
        l2_lambda: float,
        gamma: float,
        rng: np.random.Generator | None,
    ) -> None:
        from coba.policies.ridge import RidgeRegression  # avoid circular at module level

        super().__init__(arm, rng or np.random.default_rng())
        self.n_features = n_features
        self.l2_lambda = l2_lambda
        self.gamma = gamma
        self._ridge = RidgeRegression(n_features=n_features, l2_lambda=l2_lambda, gamma=gamma)

    def update(self, x: np.ndarray, reward: float, weight: float = 1.0) -> None:
        self._ridge.update(x, reward, weight)
        self.is_fitted = True

    def update_batch(
        self, x_batch: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
    ) -> None:
        self._ridge.update_batch(x_batch, y, weights)
        self.is_fitted = True

    def reset(self) -> None:
        self._ridge.reset()
        self.is_fitted = False

    @property
    def n_obs(self) -> int:
        return self._ridge.n_obs

    @property
    def beta(self) -> np.ndarray:
        """Learned coefficient vector (for inspection/debugging)."""
        return self._ridge.beta


class _LogisticBackedArmModel(BaseArmModel):
    """Shared boilerplate for arm models backed by OnlineLogisticRegression.

    Subclasses only need to implement ``score()``. All update / reset / property
    logic is provided here so it is never duplicated across LogisticUCB / LogisticTS.
    """

    def __init__(
        self,
        arm: Arm,
        n_features: int,
        l2_lambda: float,
        gamma: float,
        rng: np.random.Generator | None,
    ) -> None:
        from coba.policies.logistic import (
            OnlineLogisticRegression,
        )  # avoid circular at module level

        super().__init__(arm, rng or np.random.default_rng())
        self.n_features = n_features
        self.l2_lambda = l2_lambda
        self.gamma = gamma
        self.model = OnlineLogisticRegression(n_features, l2_lambda, gamma)

    def update(self, x: np.ndarray, reward: float, weight: float = 1.0) -> None:
        self.model.update(x, reward, weight)
        self.is_fitted = True

    def update_batch(
        self, x_batch: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
    ) -> None:
        self.model.update_batch(x_batch, y, weights)
        self.is_fitted = True

    def reset(self) -> None:
        self.model.reset()
        self.is_fitted = False

    @property
    def n_obs(self) -> int:
        return self.model.n_obs
