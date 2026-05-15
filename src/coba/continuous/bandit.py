"""ContinuousBandit: Public façade for continuous action bandits.

This module mirrors ClusterBandit API but for continuous actions, making
it a drop-in alternative for scenarios requiring real-valued decisions.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from coba.continuous.action_tree import BinaryActionTree
from coba.continuous.policy import CATSPolicy
from coba.continuous.schemas import ContinuousDecision
from coba.config import BanditConfig


class ContinuousBandit:
    """Continuous-action contextual bandit using CATS (Continuous Action Tree Sampling).

    Selects real-valued actions from [a_min, a_max] by partitioning the space into
    2^depth leaves, each with a LinUCB model. This is the continuous analog of
    ClusterBandit's discrete arm selection.

    Args:
        a_min: Lower bound of action space.
        a_max: Upper bound of action space.
        n_features: Dimensionality of context vectors.
        config: BanditConfig object. Backward-compatible kwargs override config fields.
        depth: Binary tree depth (2^depth leaves). Default 6 → 64 leaves.
        alpha: LinUCB exploration parameter.
        l2_lambda: L2 regularization for ridge regression.
        gamma: Discount factor for non-stationary environments.
        seed: Random seed.

    Attributes:
        is_fitted: True if the bandit has been trained on data.
        n_leaves: Number of leaves (2^depth).

    Example::

        bandit = ContinuousBandit(a_min=0.50, a_max=5.00, n_features=8, depth=6)
        decision = bandit.decide(context)
        bandit.update(context, decision.chosen_action, reward, decision.propensity)
    """

    def __init__(
        self,
        a_min: float,
        a_max: float,
        n_features: int,
        config: BanditConfig | None = None,
        depth: int | None = None,
        alpha: float | None = None,
        l2_lambda: float | None = None,
        gamma: float | None = None,
        seed: int | None = None,
    ) -> None:
        # Build effective config: start from provided config (or defaults),
        # then overlay any explicitly passed kwargs.
        base = config or BanditConfig()
        cfg = BanditConfig(
            cats_a_min=a_min,
            cats_a_max=a_max,
            cats_depth=depth if depth is not None else base.cats_depth,
            alpha=alpha if alpha is not None else base.alpha,
            l2_lambda=l2_lambda if l2_lambda is not None else base.l2_lambda,
            gamma=gamma if gamma is not None else base.gamma,
            seed=seed if seed is not None else base.seed,
        )
        self._config = cfg
        self.a_min = float(a_min)
        self.a_max = float(a_max)
        self.n_features = int(n_features)

        # Create the action tree
        self._tree = BinaryActionTree(
            a_min=self.a_min,
            a_max=self.a_max,
            depth=cfg.cats_depth,
        )

        # Create the CATS policy
        self._policy = CATSPolicy(
            tree=self._tree,
            n_features=n_features,
            alpha=cfg.alpha,
            l2_lambda=cfg.l2_lambda,
            gamma=cfg.gamma,
            seed=cfg.seed,
        )

    # ---- Core Online API ----

    def decide(self, context: np.ndarray) -> ContinuousDecision:
        """Select an action for the given context.

        Args:
            context: Feature vector, shape (n_features,).

        Returns:
            ContinuousDecision with chosen_action and propensity.
        """
        x = np.asarray(context, dtype=np.float64)
        if x.ndim != 1 or x.shape[0] != self.n_features:
            raise ValueError(
                f"context shape mismatch: expected ({self.n_features},), " f"got {x.shape}"
            )
        return self._policy.decide(x)

    def decide_batch(self, contexts: np.ndarray) -> list[ContinuousDecision]:
        """Vectorized batch decision for multiple contexts.

        Args:
            contexts: Feature matrix, shape (n_samples, n_features).

        Returns:
            List of ContinuousDecision objects, one per row.
        """
        contexts_arr = np.asarray(contexts, dtype=np.float64)
        if contexts_arr.ndim != 2 or contexts_arr.shape[1] != self.n_features:
            raise ValueError(
                f"contexts shape mismatch: expected (?, {self.n_features}), "
                f"got {contexts_arr.shape}"
            )
        return [self._policy.decide(x) for x in contexts_arr]

    def update(
        self,
        context: np.ndarray,
        action: float,
        reward: float,
        propensity: float = 1.0,
    ) -> None:
        """Update the bandit with an observed (context, action, reward) tuple.

        Args:
            context: Feature vector, shape (n_features,).
            action: Selected action.
            reward: Observed scalar reward.
            propensity: Probability density under the logging policy. Default 1.0.
        """
        x = np.asarray(context, dtype=np.float64)
        if x.ndim != 1 or x.shape[0] != self.n_features:
            raise ValueError(
                f"context shape mismatch: expected ({self.n_features},), " f"got {x.shape}"
            )
        self._policy.update(x, float(action), float(reward), float(propensity))

    def update_batch(
        self,
        contexts: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        propensities: np.ndarray | None = None,
    ) -> None:
        """Batch update from parallel arrays.

        Args:
            contexts: Feature matrix, shape (n_samples, n_features).
            actions: Action array, shape (n_samples,).
            rewards: Reward array, shape (n_samples,).
            propensities: Propensity array. None → uniform 1.0.
        """
        if len(rewards) == 0:
            return
        self._policy.fit_batch(
            np.asarray(contexts, dtype=np.float64),
            np.asarray(actions, dtype=np.float64),
            np.asarray(rewards, dtype=np.float64),
            propensities=(
                None if propensities is None else np.asarray(propensities, dtype=np.float64)
            ),
        )

    def fit_offline(
        self,
        contexts: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        propensities: np.ndarray | None = None,
    ) -> ContinuousBandit:
        """Bootstrap the bandit from historical logs.

        Args:
            contexts: Feature matrix, shape (n_samples, n_features).
            actions: Historical actions, shape (n_samples,).
            rewards: Historical rewards, shape (n_samples,).
            propensities: Propensity of the logging policy. None → uniform.

        Returns:
            self (for method chaining).
        """
        if propensities is None:
            propensities = np.ones(len(rewards), dtype=np.float64)
            logger.info("No propensities provided — assuming uniform logging policy")

        self._policy.fit_batch(
            np.asarray(contexts, dtype=np.float64),
            np.asarray(actions, dtype=np.float64),
            np.asarray(rewards, dtype=np.float64),
            propensities=np.asarray(propensities, dtype=np.float64),
        )
        return self

    # ---- Monitoring ----

    def get_stats(self) -> dict[int, dict]:
        """Return per-leaf statistics for monitoring dashboards.

        Returns:
            Dictionary of leaf_index → stats dict (n_obs, leaf bounds, etc.).
        """
        return self._policy.get_leaf_stats()

    # ---- Properties ----

    @property
    def is_fitted(self) -> bool:
        """True if the bandit has been trained on data."""
        return self._policy.is_fitted

    @property
    def n_leaves(self) -> int:
        """Number of leaves in the action tree."""
        return self._tree.n_leaves

    def __repr__(self) -> str:
        return (
            f"ContinuousBandit(a_min={self.a_min}, a_max={self.a_max}, "
            f"n_features={self.n_features}, depth={self._tree.depth}, "
            f"n_leaves={self.n_leaves})"
        )
