"""
Abstract base class for all bandit arm models in coba.

Each arm maintains its own statistical model (posterior, regression, counters).
This is the "one model per arm" paradigm, consistent with the OvR approach in
contextualbandits and the arm_to_model dict in mabwiser.
"""

import abc
from dataclasses import dataclass

import numpy as np

from coba.types import Arm


@dataclass
class ArmStats:
    """Lightweight statistics tracker for a single arm.

    Tracks pull count and running mean reward for monitoring purposes.
    Not used in the learning update itself (each arm model handles that internally).
    """

    arm: Arm
    n_pulls: int = 0
    reward_sum: float = 0.0

    @property
    def mean_reward(self) -> float:
        """Running mean reward, returns 0.0 if arm was never pulled."""
        if self.n_pulls == 0:
            return 0.0
        return self.reward_sum / self.n_pulls

    def record(self, reward: float) -> None:
        self.n_pulls += 1
        self.reward_sum += reward


class BaseArmModel(abc.ABC):
    """Abstract per-arm model.

    Every arm in the bandit has one instance of a BaseArmModel subclass.
    The model is responsible for:
      - Maintaining its own sufficient statistics (A, Xty, beta, counts, etc.)
      - Providing a `score(x)` method used by the bandit to rank arms.
      - Updating itself given new (context, reward) observations via `update()`.
    """

    def __init__(self, arm: Arm, rng: np.random.Generator) -> None:
        self.arm = arm
        self.rng = rng
        self.is_fitted: bool = False

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
