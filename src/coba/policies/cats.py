"""CATS leaf model: LinUCB model per action tree leaf.

CATSLeafModel wraps a per-leaf LinUCB instance. A tree of these models
constitutes the complete CATS policy (see continuous/policy.py).

This module is intentionally minimal — it delegates all LinUCB math to
the parent LinUCBArmModel, enforcing DRY principle.
"""

import numpy as np

from coba.policies.linucb import LinUCBArmModel
from coba.continuous.action_tree import ActionLeaf
from coba.types import Arm


class CATSLeafModel(LinUCBArmModel):
    """Per-leaf LinUCB model for CATS action tree.

    This is a thin wrapper around LinUCBArmModel that ties each model to
    an ActionLeaf. The 'arm' identifier is the leaf index (int).

    Args:
        leaf: The ActionLeaf this model belongs to.
        n_features: Context vector dimensionality.
        alpha: LinUCB exploration parameter.
        l2_lambda: L2 regularization strength.
        gamma: Discount factor for non-stationary environments.
        rng: NumPy random generator.
    """

    def __init__(
        self,
        leaf: ActionLeaf,
        n_features: int,
        alpha: float = 1.0,
        l2_lambda: float = 1.0,
        gamma: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        # Use leaf index as the arm identifier (required by LinUCBArmModel)
        arm: Arm = leaf.index
        super().__init__(
            arm=arm,
            n_features=n_features,
            alpha=alpha,
            l2_lambda=l2_lambda,
            gamma=gamma,
            rng=rng,
        )
        self.leaf = leaf

    def score_decomposed(self, x: np.ndarray) -> tuple[float, float]:
        """Return (mean_estimate, confidence_width) separately.

        Inherited from LinUCBArmModel but documented here for clarity.

        Args:
            x: Context vector, shape (n_features,).

        Returns:
            Tuple of (expected_reward, ucb_width).
        """
        return super().score_decomposed(x)
