"""CATSPolicy: Continuous Action Tree Sampling algorithm core.

Reference: Wen et al., "Efficient Exploration for Continuous Action Spaces",
arXiv:1902.01520, ICML 2020 + Vowpal Wabbit's --cats implementation.

Core algorithm:
  1. Score all tree leaves using their LinUCB models
  2. Select the best-scoring leaf
  3. Sample action uniformly within [midpoint - h, midpoint + h], clamped to [a_min, a_max]
  4. Return action + propensity (probability density under CATS policy)

Update:
  - Assign observation to the leaf containing the action
  - Update that leaf's LinUCB model with IPS weighting
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from coba.continuous.action_tree import BinaryActionTree
from coba.continuous.schemas import ContinuousDecision
from coba.policies.cats import CATSLeafModel


class CATSPolicy:
    """Continuous Action Tree Sampling policy.

    Partitions action space [a_min, a_max] into 2^depth leaves, each with
    a LinUCB model. Learns the value-maximizing action region per context.

    Args:
        tree: BinaryActionTree partitioning the action space.
        n_features: Dimensionality of context vectors.
        alpha: LinUCB exploration parameter.
        l2_lambda: L2 regularization for ridge regression.
        gamma: Exponential discount factor (1.0 = stationary).
        seed: Random seed for action sampling.
    """

    def __init__(
        self,
        tree: BinaryActionTree,
        n_features: int,
        alpha: float = 1.0,
        l2_lambda: float = 1.0,
        gamma: float = 1.0,
        seed: int = 42,
    ) -> None:
        self._tree = tree
        self._n_features = n_features
        self._alpha = alpha
        self._l2_lambda = l2_lambda
        self._gamma = gamma
        self._rng = np.random.default_rng(seed)

        # Initialize one LinUCB model per leaf
        self._leaf_models: dict[int, CATSLeafModel] = {
            leaf.index: CATSLeafModel(
                leaf=leaf,
                n_features=n_features,
                alpha=alpha,
                l2_lambda=l2_lambda,
                gamma=gamma,
                rng=self._rng,
            )
            for leaf in tree.leaves()
        }

        self._total_pulls: int = 0
        self.is_fitted: bool = False

    # ---- Private helpers ----

    # ---- Public API ----

    def decide(self, context: np.ndarray) -> ContinuousDecision:
        """Select an action for the given context using CATS.

        Algorithm:
          1. Score all leaves
          2. Find best-scoring leaf
          3. Sample action uniformly within [m - h, m + h]
          4. Compute propensity (probability density)

        Args:
            context: Feature vector, shape (n_features,).

        Returns:
            ContinuousDecision with chosen_action, propensity, etc.
        """
        x = np.asarray(context, dtype=np.float64)
        if x.ndim != 1 or x.shape[0] != self._n_features:
            raise ValueError(
                f"context shape mismatch: expected ({self._n_features},), got {x.shape}"
            )
        if not np.all(np.isfinite(x)):
            raise ValueError("context contains non-finite values")

        # Score all leaves
        all_scores = self.score_all_leaves(x)

        # Find best leaf
        best_leaf_idx = max(all_scores, key=all_scores.get)  # type: ignore[arg-type]
        best_leaf = self._tree.leaves()[best_leaf_idx]

        # Decompose best leaf's score
        mean_est, conf_width = self._leaf_models[best_leaf_idx].score_decomposed(x)

        # Sample action uniformly within bandwidth window
        window_lo = max(best_leaf.midpoint - self._tree.bandwidth, self._tree.a_min)
        window_hi = min(best_leaf.midpoint + self._tree.bandwidth, self._tree.a_max)
        chosen_action = self._rng.uniform(window_lo, window_hi)

        # Compute propensity: uniform density over effective window width
        effective_width = window_hi - window_lo
        propensity = 1.0 / effective_width if effective_width > 0 else 1.0

        self._total_pulls += 1

        return ContinuousDecision(
            chosen_action=float(chosen_action),
            propensity=float(propensity),
            leaf_index=best_leaf_idx,
            leaf_lo=best_leaf.lo,
            leaf_hi=best_leaf.hi,
            leaf_scores=all_scores,
            mean_estimate=float(mean_est),
            confidence_width=float(conf_width),
        )

    def score_all_leaves(self, context: np.ndarray) -> dict[int, float]:
        """Score all leaves for a given context.

        Args:
            context: Feature vector, shape (n_features,).

        Returns:
            Dictionary mapping leaf_index → LinUCB score.
        """
        x = np.asarray(context, dtype=np.float64)
        return {leaf_idx: float(model.score(x)) for leaf_idx, model in self._leaf_models.items()}

    def update(
        self,
        context: np.ndarray,
        action: float,
        reward: float,
        propensity: float = 1.0,
    ) -> None:
        """Update the policy with an observed (context, action, reward) tuple.

        Args:
            context: Feature vector, shape (n_features,).
            action: Selected action (between a_min and a_max).
            reward: Observed scalar reward.
            propensity: Probability density p(action | context) under the
                       logging policy (e.g., 1/(2h) for CATS). Used for IPS.
        """
        x = np.asarray(context, dtype=np.float64)
        if not np.isfinite(reward):
            raise ValueError(f"reward must be finite, got {reward}")
        if propensity <= 0 or not np.isfinite(propensity):
            raise ValueError(f"propensity must be positive and finite, got {propensity}")

        # Find the leaf containing this action
        leaf = self._tree.leaf_for_action(action)

        # Compute IPS weight: 1 / propensity (clipped for stability)
        ips_weight = 1.0 / max(propensity, 1e-4)
        ips_weight = float(np.clip(ips_weight, 0.0, 10.0))

        # Update the leaf's model
        self._leaf_models[leaf.index].update(x, reward, weight=ips_weight)
        self.is_fitted = True
        self._total_pulls += 1

    def fit_batch(
        self,
        contexts: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        propensities: np.ndarray | None = None,
    ) -> CATSPolicy:
        """Batch update from parallel arrays (for offline training).

        Args:
            contexts: Feature matrix, shape (n_samples, n_features).
            actions: Action array, shape (n_samples,).
            rewards: Reward array, shape (n_samples,).
            propensities: Propensity array, shape (n_samples,). None → uniform 1.0.

        Returns:
            self (for method chaining).
        """
        contexts_arr = np.asarray(contexts, dtype=np.float64)
        actions_arr = np.asarray(actions, dtype=np.float64)
        rewards_arr = np.asarray(rewards, dtype=np.float64)

        if contexts_arr.ndim != 2 or contexts_arr.shape[1] != self._n_features:
            raise ValueError(
                f"contexts shape mismatch: expected (?, {self._n_features}), "
                f"got {contexts_arr.shape}"
            )
        if len(contexts_arr) != len(actions_arr) or len(actions_arr) != len(rewards_arr):
            raise ValueError("contexts, actions, and rewards must have the same length")

        # Default propensities to 1.0
        if propensities is None:
            propensities_arr = np.ones(len(rewards_arr), dtype=np.float64)
        else:
            propensities_arr = np.asarray(propensities, dtype=np.float64)
            if len(propensities_arr) != len(rewards_arr):
                raise ValueError("propensities length must match rewards")

        # Compute IPS weights
        weights = np.clip(1.0 / np.maximum(propensities_arr, 1e-4), 0.0, 10.0)

        # Assign each observation to its leaf and accumulate per-leaf batches
        # Type is list[...] tuples at build time; np.array() converts before use.
        leaf_batches: dict[int, tuple[list[float], list[float], list[float]]] = {}
        for i, (ctx, action, reward, weight) in enumerate(
            zip(contexts_arr, actions_arr, rewards_arr, weights)
        ):
            leaf = self._tree.leaf_for_action(action)
            leaf_idx = leaf.index
            if leaf_idx not in leaf_batches:
                leaf_batches[leaf_idx] = ([], [], [])
            leaf_batches[leaf_idx][0].append(ctx)
            leaf_batches[leaf_idx][1].append(reward)
            leaf_batches[leaf_idx][2].append(weight)

        # Update each leaf's model with its accumulated batch
        for leaf_idx, (ctx_list, rew_list, wt_list) in leaf_batches.items():
            ctx_batch = np.array(ctx_list, dtype=np.float64)
            rew_batch = np.array(rew_list, dtype=np.float64)
            wt_batch = np.array(wt_list, dtype=np.float64)
            self._leaf_models[leaf_idx].update_batch(ctx_batch, rew_batch, wt_batch)

        self._total_pulls += len(contexts_arr)
        self.is_fitted = True
        logger.info(
            "CATSPolicy batch fitted with {n} observations across {k} leaves",
            n=len(rewards_arr),
            k=len(leaf_batches),
        )
        return self

    # ---- Monitoring & Utilities ----

    def get_leaf_stats(self) -> dict[int, dict[str, float | int]]:
        """Return per-leaf statistics (n_obs, mean_reward, etc.).

        Returns:
            Dictionary mapping leaf_index → stats dict.
        """
        stats = {}
        for leaf_idx, model in self._leaf_models.items():
            stats[leaf_idx] = {
                "n_obs": model.n_obs,
                "leaf_lo": model.leaf.lo,
                "leaf_hi": model.leaf.hi,
                "midpoint": model.leaf.midpoint,
            }
        return stats

    def reset(self) -> None:
        """Reset all leaf models to their prior state."""
        for model in self._leaf_models.values():
            model.reset()
        self.is_fitted = False
        self._total_pulls = 0

    @property
    def n_leaves(self) -> int:
        """Number of leaves in the action tree."""
        return self._tree.n_leaves

    def __repr__(self) -> str:
        return (
            f"CATSPolicy(tree={self._tree}, n_features={self._n_features}, "
            f"alpha={self._alpha}, n_leaves={self.n_leaves})"
        )
