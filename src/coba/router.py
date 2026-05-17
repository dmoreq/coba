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

from coba.config import BanditConfig
from coba.policies.base import BaseArmModel
from coba.policies.ridge import RidgeRegression
from coba.types import Arm, PolicyType


def _build_model_for_arm(
    arm: Arm,
    cfg: BanditConfig,
    n_features: int,
    rng: np.random.Generator,
    shared_ridge: "Any | None" = None,
    neural_backbone: "Any | None" = None,
) -> BaseArmModel:
    """Registry-based factory: return one BaseArmModel for a single arm.

    Adding a new policy only requires registering an entry in ``_POLICY_REGISTRY``
    at module load time — no ``match`` statement surgery needed.
    """
    # Lazy imports keep module-level load time low and avoid circular imports.
    from coba.policies.gp_ucb import GPUCBArmModel
    from coba.policies.lin_ucb_hybrid import LinUCBHybridArmModel
    from coba.policies.linucb import LinUCBArmModel
    from coba.policies.linucb_sw import SlidingWindowLinUCBArmModel
    from coba.policies.lin_ts import LinTSArmModel
    from coba.policies.logistic import LogisticTSArmModel, LogisticUCBArmModel
    from coba.policies.neural_linear import NeuralLinearArmModel
    from coba.policies.sklearn_models import (
        BootstrappedTSArmModel,
        BootstrappedUCBArmModel,
        EpsilonGreedyArmModel,
    )
    from coba.policies.softmax import SoftmaxArmModel
    from coba.policies.thompson import ThompsonArmModel
    from coba.policies.tree_ensemble import RandomForestTSArmModel, RandomForestUCBArmModel
    from coba.policies.ucb1 import UCB1ArmModel

    p = cfg.policy

    if p == PolicyType.NEURAL_LINEAR:
        if neural_backbone is None:
            raise ValueError("neural_backbone must be provided for NEURAL_LINEAR policy")
        return NeuralLinearArmModel(
            arm,
            backbone=neural_backbone,
            v_sq=cfg.v_sq,
            l2_lambda=cfg.l2_lambda,
            gamma=cfg.gamma,
            rng=rng,
        )
    if p == PolicyType.LIN_UCB_HYBRID:
        if shared_ridge is None:
            raise ValueError("shared_ridge must be provided for LIN_UCB_HYBRID policy")
        return LinUCBHybridArmModel(
            arm,
            n_shared=cfg.n_shared_features,
            n_arm=n_features - cfg.n_shared_features,
            shared_ridge=shared_ridge,
            alpha=cfg.alpha,
            l2_lambda=cfg.l2_lambda,
            rng=rng,
            gamma=cfg.gamma,
        )
    if p == PolicyType.LIN_UCB:
        return LinUCBArmModel(
            arm,
            n_features,
            alpha=cfg.alpha,
            l2_lambda=cfg.l2_lambda,
            rng=rng,
            gamma=cfg.gamma,
        )
    if p == PolicyType.LIN_TS:
        return LinTSArmModel(
            arm,
            n_features,
            v_sq=cfg.v_sq,
            l2_lambda=cfg.l2_lambda,
            rng=rng,
            gamma=cfg.gamma,
        )
    if p == PolicyType.THOMPSON:
        return ThompsonArmModel(arm, rng=rng)
    if p == PolicyType.UCB1:
        return UCB1ArmModel(arm, alpha=cfg.alpha, rng=rng)
    if p == PolicyType.EPSILON_GREEDY:
        return EpsilonGreedyArmModel(
            arm,
            rng=rng,
            base_estimator=cfg.base_estimator,
            epsilon=cfg.epsilon,
        )
    if p == PolicyType.BOOTSTRAPPED_TS:
        return BootstrappedTSArmModel(
            arm,
            rng=rng,
            base_estimator=cfg.base_estimator,
            n_bootstraps=cfg.n_bootstraps,
        )
    if p == PolicyType.BOOTSTRAPPED_UCB:
        return BootstrappedUCBArmModel(
            arm,
            rng=rng,
            base_estimator=cfg.base_estimator,
            n_bootstraps=cfg.n_bootstraps,
        )
    if p == PolicyType.LOGISTIC_UCB:
        return LogisticUCBArmModel(
            arm,
            n_features,
            alpha=cfg.alpha,
            l2_lambda=cfg.l2_lambda,
            rng=rng,
            gamma=cfg.gamma,
        )
    if p == PolicyType.LOGISTIC_TS:
        return LogisticTSArmModel(
            arm,
            n_features,
            v_sq=cfg.v_sq,
            l2_lambda=cfg.l2_lambda,
            rng=rng,
            gamma=cfg.gamma,
        )
    if p == PolicyType.GP_UCB:
        return GPUCBArmModel(
            arm,
            beta=cfg.gp_beta,
            length_scale=cfg.gp_length_scale,
            noise_var=cfg.gp_noise_var,
            max_obs=cfg.gp_max_obs,
            rng=rng,
        )
    if p == PolicyType.SOFTMAX:
        return SoftmaxArmModel(
            arm,
            n_features,
            tau=cfg.softmax_tau,
            l2_lambda=cfg.l2_lambda,
            gamma=cfg.gamma,
            rng=rng,
        )
    if p == PolicyType.LIN_UCB_SW:
        return SlidingWindowLinUCBArmModel(
            arm,
            n_features,
            window_size=cfg.linucb_sw_window,
            alpha=cfg.alpha,
            l2_lambda=cfg.l2_lambda,
            rng=rng,
        )
    if p == PolicyType.RANDOM_FOREST_UCB:
        return RandomForestUCBArmModel(
            arm,
            rng=rng,
            alpha=cfg.alpha,
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            min_samples_leaf=cfg.rf_min_samples_leaf,
            max_obs=cfg.rf_max_obs,
            min_uncertainty=cfg.rf_min_uncertainty,
        )
    if p == PolicyType.RANDOM_FOREST_TS:
        return RandomForestTSArmModel(
            arm,
            rng=rng,
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            min_samples_leaf=cfg.rf_min_samples_leaf,
            max_obs=cfg.rf_max_obs,
            min_uncertainty=cfg.rf_min_uncertainty,
        )
    raise ValueError(f"Unsupported policy: {p}")


def _build_arm_models(
    arms: list[Arm],
    cfg: BanditConfig,
    n_features: int,
    rng: np.random.Generator,
    shared_ridge: "Any | None" = None,
    neural_backbone: "Any | None" = None,
) -> dict[Arm, BaseArmModel]:
    """Build one model per arm using the registry factory."""
    return {
        arm: _build_model_for_arm(arm, cfg, n_features, rng, shared_ridge, neural_backbone)
        for arm in arms
    }


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
        n_features: int,
        config: BanditConfig | None = None,
    ) -> None:
        cfg = config or BanditConfig()

        if cfg.n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")
        if not arms:
            raise ValueError("arms list cannot be empty.")

        self.arms = list(arms)
        self.n_clusters = cfg.n_clusters
        self.policy = cfg.policy
        self.n_features = n_features
        self._cfg = cfg

        self._rng = np.random.default_rng(cfg.seed)

        # Cluster model: KMeans assigns each context to a cluster
        if cfg.use_minibatch:
            self._kmeans: KMeans | MiniBatchKMeans = MiniBatchKMeans(
                n_clusters=cfg.n_clusters, random_state=cfg.seed, n_init=3
            )
        else:
            self._kmeans = KMeans(n_clusters=cfg.n_clusters, random_state=cfg.seed, n_init=10)

        # StandardScaler for context normalization
        self._scaler: StandardScaler | None = StandardScaler() if cfg.scale_contexts else None

        # For NEURAL_LINEAR each cluster gets its own NeuralLinearBackbone.
        if cfg.policy == PolicyType.NEURAL_LINEAR:
            from coba.policies.neural_linear import NeuralLinearBackbone

            self._neural_backbones: list[NeuralLinearBackbone] | None = [
                NeuralLinearBackbone(
                    n_features=n_features,
                    embedding_dim=cfg.neural_embedding_dim,
                    hidden_sizes=cfg.neural_hidden_sizes,
                    retrain_freq=cfg.neural_retrain_freq,
                    seed=cfg.seed + c,
                )
                for c in range(cfg.n_clusters)
            ]
        else:
            self._neural_backbones = None

        # For LIN_UCB_HYBRID each cluster gets its own SharedRidge.
        if cfg.policy == PolicyType.LIN_UCB_HYBRID:
            if cfg.n_shared_features <= 0 or cfg.n_shared_features >= n_features:
                raise ValueError(
                    f"LIN_UCB_HYBRID requires 0 < n_shared_features < n_features, "
                    f"got n_shared_features={cfg.n_shared_features}, n_features={n_features}"
                )
            self._shared_ridges: list[RidgeRegression] | None = [
                RidgeRegression(
                    n_features=cfg.n_shared_features,
                    l2_lambda=cfg.l2_lambda,
                    gamma=cfg.gamma,
                )
                for _ in range(cfg.n_clusters)
            ]
        else:
            self._shared_ridges = None

        self._cluster_bandits: list[dict[Arm, BaseArmModel]] = [
            _build_arm_models(
                arms,
                cfg,
                n_features,
                self._rng,
                shared_ridge=self._shared_ridges[c] if self._shared_ridges is not None else None,
                neural_backbone=(
                    self._neural_backbones[c] if self._neural_backbones is not None else None
                ),
            )
            for c in range(cfg.n_clusters)
        ]

        self._total_pulls: int = 0
        self.is_fitted: bool = False

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
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_batch(
        self,
        contexts: np.ndarray,
        decisions: np.ndarray,
        rewards: np.ndarray,
        weights: np.ndarray | None,
        fit_scaler: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Cast inputs and optionally fit/apply the scaler.

        Returns:
            Tuple of (scaled_contexts, decisions, rewards, weights).
        """
        ctx = np.asarray(contexts, dtype=np.float64)
        dec = np.asarray(decisions)
        rew = np.asarray(rewards, dtype=np.float64)
        wts = (
            np.ones(len(rew), dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )

        if self._scaler is not None:
            if fit_scaler:
                self._scaler.fit(ctx)
            scaled = self._scaler.transform(ctx)
        else:
            scaled = ctx
        return scaled, dec, rew, wts

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
        scaled, decisions, rewards, weights = self._prepare_batch(
            contexts, decisions, rewards, weights, fit_scaler=True
        )

        # Fit KMeans and assign each sample to its cluster
        self._kmeans.fit(scaled)
        cluster_labels: np.ndarray = self._kmeans.labels_

        # Reset all cluster arm models before full refit
        for cluster_bandit in self._cluster_bandits:
            for model in cluster_bandit.values():
                model.reset()
        if self._shared_ridges is not None:
            for sr in self._shared_ridges:
                sr.reset()
        if self._neural_backbones is not None:
            for nb in self._neural_backbones:
                nb.reset()

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

        scaled, decisions, rewards, weights = self._prepare_batch(
            contexts, decisions, rewards, weights, fit_scaler=False
        )

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

        # After the first online update, attempt to mark as fitted so that
        # subsequent decide() calls skip the cold-start path and use real scores.
        # KMeans requires at least n_clusters samples, so we keep a small buffer
        # of (context, arm, reward, weight) tuples until we have enough.
        if not self.is_fitted:
            if not hasattr(self, "_pending_updates"):
                self._pending_updates: list[tuple] = []
            self._pending_updates.append((scaled_ctx.copy(), arm, reward, weight))
            if len(self._pending_updates) >= self.n_clusters:
                ctxs = np.array([p[0] for p in self._pending_updates])
                if self._scaler is not None and not hasattr(self._scaler, "mean_"):
                    self._scaler.fit(ctxs)
                    ctxs = self._scaler.transform(ctxs)
                self._kmeans.fit(ctxs)
                self.is_fitted = True
                self._pending_updates = []

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

        import dataclasses

        effective_gamma = gamma if gamma is not None else self._cfg.gamma
        # Build a per-arm config that may override gamma
        arm_cfg = dataclasses.replace(self._cfg, gamma=effective_gamma)
        self.arms.append(arm)

        for c_idx, cluster_bandit in enumerate(self._cluster_bandits):
            cluster_shared_ridge = (
                self._shared_ridges[c_idx] if self._shared_ridges is not None else None
            )
            cluster_neural_backbone = (
                self._neural_backbones[c_idx] if self._neural_backbones is not None else None
            )
            if warm_start_from is not None and warm_start_from in cluster_bandit:
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
                cluster_bandit[arm] = _build_model_for_arm(
                    arm,
                    arm_cfg,
                    self.n_features,
                    self._rng,
                    shared_ridge=cluster_shared_ridge,
                    neural_backbone=cluster_neural_backbone,
                )

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

    def reset_arm(self, arm: Arm) -> None:
        """Reset an arm's model across all clusters to its prior state.

        Used by drift detection to discard stale knowledge after a reward
        distribution shift is detected for that arm.

        Args:
            arm: Arm whose models should be reset.
        """
        if arm not in self.arms:
            raise ValueError(f"Arm '{arm}' not found.")
        for cluster_bandit in self._cluster_bandits:
            if arm in cluster_bandit:
                cluster_bandit[arm].reset()
        logger.debug("Reset arm {arm} across all clusters after drift detection", arm=arm)

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
