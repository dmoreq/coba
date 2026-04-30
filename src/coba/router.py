"""
KMeans-based Cluster Router for contextual bandits.

Design rationale:
  Context spaces are rarely uniform — different regions of the feature space
  may exhibit very different reward patterns. Rather than training one global
  bandit that must approximate all behaviors simultaneously, we:
    1. Cluster the context space into K behaviorally distinct groups.
    2. Train one independent bandit per cluster.
    3. At prediction time, assign the context to its nearest cluster and query
       that cluster's bandit.

  This allows each cluster's model to specialize on its local context region
  without hand-coding those regions explicitly.

  Key improvements over a naive single-bandit approach:
    - Supports incremental (partial_fit) cluster reassignment via MiniBatchKMeans.
    - Arms are managed (add/remove) across ALL cluster bandits atomically.
    - Warm-start copies the nearest trained arm's model to cold arms.

Usage:
  router = ClusterRouter(arms=["a", "b", "c"], n_clusters=5, policy="linucb")
  router.fit(contexts_matrix, decisions_array, rewards_array)
  arm = router.predict(context_vector)
  router.update(context_vector, chosen_arm, reward, weight)
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from loguru import logger
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from coba.policies.base import BaseArmModel
from coba.types import Arm, PolicyType


def _build_arm_models(
    arms: list[Arm],
    policy: PolicyType,
    n_features: int,
    rng: np.random.Generator,
    alpha: float,
    v_sq: float,
    l2_lambda: float,
    base_estimator: "Any | None" = None,
    n_bootstraps: int = 10,
    epsilon: float = 0.1,
    gamma: float = 1.0,
) -> dict[Arm, BaseArmModel]:
    """Factory function: build one BaseArmModel per arm for the given policy type."""
    from coba.policies.linucb import LinUCBArmModel
    from coba.policies.lin_ts import LinTSArmModel
    from coba.policies.sklearn_models import (
        BootstrappedTSArmModel,
        BootstrappedUCBArmModel,
        EpsilonGreedyArmModel,
    )
    from coba.policies.thompson import ThompsonArmModel
    from coba.policies.ucb1 import UCB1ArmModel
    from coba.policies.logistic import LogisticUCBArmModel, LogisticTSArmModel

    models: dict[Arm, BaseArmModel] = {}
    for arm in arms:
        match policy:
            case PolicyType.LIN_UCB:
                models[arm] = LinUCBArmModel(
                    arm, n_features, alpha=alpha, l2_lambda=l2_lambda, rng=rng, gamma=gamma
                )
            case PolicyType.LIN_TS:
                models[arm] = LinTSArmModel(
                    arm, n_features, v_sq=v_sq, l2_lambda=l2_lambda, rng=rng, gamma=gamma
                )
            case PolicyType.THOMPSON:
                models[arm] = ThompsonArmModel(arm, rng=rng)
            case PolicyType.UCB1:
                models[arm] = UCB1ArmModel(arm, alpha=alpha, rng=rng)
            case PolicyType.EPSILON_GREEDY:
                models[arm] = EpsilonGreedyArmModel(
                    arm, rng=rng, base_estimator=base_estimator, epsilon=epsilon
                )
            case PolicyType.BOOTSTRAPPED_TS:
                models[arm] = BootstrappedTSArmModel(
                    arm,
                    rng=rng,
                    base_estimator=base_estimator,
                    n_bootstraps=n_bootstraps,
                )
            case PolicyType.BOOTSTRAPPED_UCB:
                models[arm] = BootstrappedUCBArmModel(
                    arm,
                    rng=rng,
                    base_estimator=base_estimator,
                    n_bootstraps=n_bootstraps,
                )
            case PolicyType.LOGISTIC_UCB:
                models[arm] = LogisticUCBArmModel(
                    arm, n_features, alpha=alpha, l2_lambda=l2_lambda, rng=rng, gamma=gamma
                )
            case PolicyType.LOGISTIC_TS:
                models[arm] = LogisticTSArmModel(
                    arm, n_features, v_sq=v_sq, l2_lambda=l2_lambda, rng=rng, gamma=gamma
                )
            case _:
                raise ValueError(f"Unsupported policy: {policy}")
    return models


class ClusterRouter:
    """Routes context vectors to one of K specialized bandits via KMeans clustering.

    Each cluster maintains its own set of per-arm models. At decision time, the
    context is assigned to the nearest cluster centroid and the corresponding
    bandit makes the arm selection.

    Args:
        arms: List of arms, e.g. ["a", "b", "c"] or [0, 1, 2].
        n_clusters: Number of context clusters. Start with 3–8.
        policy: Which learning algorithm to use per arm per cluster.
        n_features: Dimensionality of context vectors.
        alpha: Exploration parameter for LinUCB / UCB1.
        v_sq: Variance multiplier for LinTS.
        l2_lambda: L2 regularization for linear models.
        use_minibatch: If True, uses MiniBatchKMeans for faster incremental fitting.
        scale_contexts: If True, applies StandardScaler to contexts before clustering.
        seed: Random seed for reproducibility.
        base_estimator: A scikit-learn compatible estimator for non-linear policies.
        n_bootstraps: Number of bootstrapped models to maintain for bootstrapped policies.
        epsilon: Epsilon value for epsilon-greedy exploration.
    """

    def __init__(
        self,
        arms: list[Arm],
        n_clusters: int = 5,
        policy: PolicyType = PolicyType.LIN_UCB,
        n_features: int = 7,
        alpha: float = 1.0,
        v_sq: float = 1.0,
        l2_lambda: float = 1.0,
        use_minibatch: bool = True,
        scale_contexts: bool = True,
        seed: int = 42,
        base_estimator: "Any | None" = None,
        n_bootstraps: int = 10,
        epsilon: float = 0.1,
        gamma: float = 1.0,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")
        if not arms:
            raise ValueError("arms list cannot be empty.")

        self.arms = list(arms)
        self.n_clusters = n_clusters
        self.policy = policy
        self.n_features = n_features
        self.alpha = alpha
        self.v_sq = v_sq
        self.l2_lambda = l2_lambda
        self.seed = seed
        self.scale_contexts = scale_contexts
        self.base_estimator = base_estimator
        self.n_bootstraps = n_bootstraps
        self.epsilon = epsilon
        self.gamma = gamma

        self._rng = np.random.default_rng(seed)

        # Cluster model: KMeans assigns each context to a cluster
        if use_minibatch:
            self._kmeans: KMeans | MiniBatchKMeans = MiniBatchKMeans(
                n_clusters=n_clusters, random_state=seed, n_init=3
            )
        else:
            self._kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)

        # StandardScaler for context normalization
        self._scaler: StandardScaler | None = StandardScaler() if scale_contexts else None

        # One dict of {arm: ArmModel} per cluster
        self._cluster_bandits: list[dict[Arm, BaseArmModel]] = [
            _build_arm_models(
                arms,
                policy,
                n_features,
                self._rng,
                alpha,
                v_sq,
                l2_lambda,
                base_estimator=self.base_estimator,
                n_bootstraps=self.n_bootstraps,
                epsilon=self.epsilon,
                gamma=self.gamma,
            )
            for _ in range(n_clusters)
        ]

        # Total pulls per arm across the entire router (for UCB1 denominator)
        self._total_pulls: int = 0

        self.is_fitted: bool = False

        # Scoring function differs for UCB1 (needs global pull count) vs others.
        # Use named methods (not lambdas) so joblib.dump can pickle this object.
        if self.policy == PolicyType.UCB1:
            self._score_fn: Callable[[BaseArmModel, np.ndarray], float] = self._score_ucb1
        else:
            self._score_fn: Callable[[BaseArmModel, np.ndarray], float] = self._score_default

    # ------------------------------------------------------------------
    # Internal scoring helpers (named so joblib can pickle this object)
    # ------------------------------------------------------------------

    def _score_ucb1(self, model: BaseArmModel, ctx: np.ndarray) -> float:
        """Score function for UCB1 — passes the global pull count."""
        return model.score(ctx, total_pulls=self._total_pulls)  # type: ignore[call-arg]

    def _score_default(self, model: BaseArmModel, ctx: np.ndarray) -> float:
        """Score function for all contextual policies."""
        return model.score(ctx)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        contexts: np.ndarray,
        decisions: np.ndarray,
        rewards: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> "ClusterRouter":
        """Full batch fit: cluster the contexts and train each cluster's bandit.

        Args:
            contexts: Feature matrix, shape (n_samples, n_features).
            decisions: Arm chosen per sample, shape (n_samples,).
            rewards: Observed reward per sample, shape (n_samples,).
            weights: IPS importance weights, shape (n_samples,). None → all ones.
        """
        contexts = np.asarray(contexts, dtype=np.float64)
        decisions = np.asarray(decisions)
        rewards = np.asarray(rewards, dtype=np.float64)
        weights = (
            np.ones(len(rewards)) if weights is None else np.asarray(weights, dtype=np.float64)
        )

        # Scale contexts
        if self._scaler is not None:
            self._scaler.fit(contexts)
            scaled = self._scaler.transform(contexts)
        else:
            scaled = contexts

        # Fit KMeans and assign each sample to its cluster
        self._kmeans.fit(scaled)
        cluster_labels: np.ndarray = self._kmeans.labels_

        # Reset all cluster arm models before full refit
        for cluster_bandit in self._cluster_bandits:
            for model in cluster_bandit.values():
                model.reset()

        # Train each cluster's bandit on its subset of the data
        for c in range(self.n_clusters):
            mask = cluster_labels == c
            if not np.any(mask):
                logger.debug("Cluster {c} has no training samples — skipping", c=c)
                continue
            self._fit_cluster(
                cluster_idx=c,
                contexts=scaled[mask],
                decisions=decisions[mask],
                rewards=rewards[mask],
                weights=weights[mask],
            )

        self._total_pulls = len(decisions)
        self.is_fitted = True
        logger.info(
            "ClusterRouter fitted with {n} samples across {k} clusters",
            n=len(rewards),
            k=self.n_clusters,
        )
        return self

    def partial_fit(
        self,
        contexts: np.ndarray,
        decisions: np.ndarray,
        rewards: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> "ClusterRouter":
        """Online update: assign new samples to existing clusters and update models.

        Does NOT refit the KMeans — the cluster assignment is frozen from the
        initial `fit()` call. This is intentional: in production, the cluster
        structure should remain stable across incremental updates.

        Args:
            contexts: Feature matrix, shape (n_samples, n_features).
            decisions: Arm chosen per sample, shape (n_samples,).
            rewards: Observed reward per sample, shape (n_samples,).
            weights: IPS importance weights. None → all ones.
        """
        if not self.is_fitted:
            return self.fit(contexts, decisions, rewards, weights)

        contexts = np.asarray(contexts, dtype=np.float64)
        decisions = np.asarray(decisions)
        rewards = np.asarray(rewards, dtype=np.float64)
        weights = (
            np.ones(len(rewards)) if weights is None else np.asarray(weights, dtype=np.float64)
        )

        if self._scaler is not None:
            scaled = self._scaler.transform(contexts)
        else:
            scaled = contexts

        # Predict cluster assignment using existing centroids (no refit)
        cluster_labels: np.ndarray = self._kmeans.predict(scaled)

        for c in range(self.n_clusters):
            mask = cluster_labels == c
            if not np.any(mask):
                continue
            self._fit_cluster(
                cluster_idx=c,
                contexts=scaled[mask],
                decisions=decisions[mask],
                rewards=rewards[mask],
                weights=weights[mask],
            )

        self._total_pulls += len(decisions)
        return self

    def predict(self, context: np.ndarray) -> Arm:
        """Select the arm with the highest score for the given context.

        Args:
            context: Feature vector, shape (n_features,).
        Returns:
            The chosen arm identifier.
        """
        cluster_idx, scaled_ctx = self._route(context)
        arm_models = self._cluster_bandits[cluster_idx]

        best_arm: Arm | None = None
        best_score = float("-inf")

        for arm, model in arm_models.items():
            s = self._score_fn(model, scaled_ctx)
            if s > best_score:
                best_score = s
                best_arm = arm

        assert best_arm is not None, "No arms available — arms list is empty."
        return best_arm

    def score_all(self, context: np.ndarray) -> dict[Arm, float]:
        """Return scores for all arms given the context.

        Useful for debugging and for building propensity estimates.

        Args:
            context: Feature vector, shape (n_features,).
        Returns:
            Dict mapping each arm to its score.
        """
        cluster_idx, scaled_ctx = self._route(context)
        arm_models = self._cluster_bandits[cluster_idx]

        scores: dict[Arm, float] = {}
        for arm, model in arm_models.items():
            scores[arm] = self._score_fn(model, scaled_ctx)
        return scores

    def update(
        self,
        context: np.ndarray,
        arm: Arm,
        reward: float,
        weight: float = 1.0,
    ) -> None:
        """Update the model for the chosen arm in the appropriate cluster.

        This is the core online learning step, called after each decision.

        Args:
            context: Feature vector, shape (n_features,).
            arm: The arm that was chosen and applied.
            reward: Observed scalar reward, normalized to [0, 1].
            weight: IPS importance weight (1/propensity). Default 1.0.
        """
        cluster_idx, scaled_ctx = self._route(context)
        arm_models = self._cluster_bandits[cluster_idx]

        if arm not in arm_models:
            raise ValueError(f"Arm '{arm}' not found in bandit. Call add_arm() first.")

        arm_models[arm].update(scaled_ctx, reward, weight)
        self._total_pulls += 1

    def add_arm(
        self,
        arm: Arm,
        warm_start_from: Arm | None = None,
        gamma: float | None = None,
    ) -> None:
        """Dynamically add a new arm to all cluster bandits.

        If warm_start_from is provided and that arm is trained, the new arm's
        model is initialized by copying the source arm's model (warm start).
        This avoids a cold start for the new arm.

        Args:
            arm: Identifier for the new arm.
            warm_start_from: If given, copy this arm's model as the starting point.
            gamma: Discount factor for the new arm's model. Overrides the router-level
                   gamma when provided. Useful for giving a newly launched arm a higher
                   decay rate so it adapts faster to its own distribution.
        """
        if arm in self.arms:
            raise ValueError(f"Arm '{arm}' already exists.")

        effective_gamma = gamma if gamma is not None else self.gamma
        self.arms.append(arm)

        for cluster_bandit in self._cluster_bandits:
            if warm_start_from is not None and warm_start_from in cluster_bandit:
                # Warm start: copy the model from the source arm
                source_model = cluster_bandit[warm_start_from]
                new_model = source_model.clone()
                new_model.arm = arm
                cluster_bandit[arm] = new_model
                logger.debug(
                    "Warm started arm {arm} from {source} in cluster",
                    arm=arm,
                    source=warm_start_from,
                )
            else:
                # Cold start: fresh model with optional per-arm gamma
                new_models = _build_arm_models(
                    [arm],
                    self.policy,
                    self.n_features,
                    self._rng,
                    self.alpha,
                    self.v_sq,
                    self.l2_lambda,
                    base_estimator=self.base_estimator,
                    n_bootstraps=self.n_bootstraps,
                    epsilon=self.epsilon,
                    gamma=effective_gamma,
                )
                cluster_bandit[arm] = new_models[arm]

    def remove_arm(self, arm: Arm) -> None:
        """Remove an arm from all cluster bandits.

        Args:
            arm: Identifier of the arm to remove.
        """
        if arm not in self.arms:
            raise ValueError(f"Arm '{arm}' not found.")
        if len(self.arms) <= 1:
            raise ValueError("Cannot remove the last arm — bandit must have at least 1 arm.")

        self.arms.remove(arm)
        for cluster_bandit in self._cluster_bandits:
            cluster_bandit.pop(arm, None)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _route(self, context: np.ndarray) -> tuple[int, np.ndarray]:
        """Scale context and return (cluster_index, scaled_context).

        Args:
            context: Raw feature vector, shape (n_features,).
        Returns:
            Tuple of (cluster_index, scaled_context).
        """
        ctx = np.asarray(context, dtype=np.float64).reshape(1, -1)

        if self._scaler is not None:
            if not hasattr(self._scaler, "mean_"):
                # Scaler not yet fitted — return cluster 0 and raw context
                return 0, ctx.ravel()
            ctx = self._scaler.transform(ctx)

        if not self.is_fitted:
            return 0, ctx.ravel()

        cluster_idx = int(self._kmeans.predict(ctx)[0])
        return cluster_idx, ctx.ravel()

    def _fit_cluster(
        self,
        cluster_idx: int,
        contexts: np.ndarray,
        decisions: np.ndarray,
        rewards: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        """Update all arm models in a specific cluster with the provided data."""
        cluster_bandit = self._cluster_bandits[cluster_idx]
        for arm, model in cluster_bandit.items():
            arm_mask = decisions == arm
            if not np.any(arm_mask):
                continue
            arm_contexts = contexts[arm_mask]
            arm_rewards = rewards[arm_mask]
            arm_weights = weights[arm_mask]
            model.update_batch(arm_contexts, arm_rewards, arm_weights)
