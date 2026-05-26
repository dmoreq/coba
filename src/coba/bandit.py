"""
ClusterBandit — Main public façade for the coba library.

This class provides a high-level API for contextual bandits with cluster routing.
It wraps a ClusterRouter and translates between generic inputs (numpy arrays)
and internal operations.

Typical flow:
  # 1. Create bandit
  bandit = ClusterBandit(
      arms=[1.0, 1.1, 1.2, 1.5],
      policy=PolicyType.LIN_UCB,
      n_clusters=5,
      n_features=7,
  )

  # 2. (Optional) Bootstrap from historical logs
  bandit.fit_offline(contexts_matrix, decisions, rewards, propensities)

  # 3. Online serving
  context = np.array([50.0, 10.0, 100.0, 5.0, 8.0, 1.0, 5.0])
  decision = bandit.decide(context)           # → BanditDecision(chosen_arm=1.2, ...)

  # 4. Collect reward and update
  bandit.update(
      context=context,
      arm=decision.chosen_arm,
      reward=0.7,                            # normalized reward
      propensity=0.8,                        # logging policy probability for this arm
  )
"""

from typing import Any

import numpy as np
from loguru import logger

from coba.config import BanditConfig
from coba.drift import PageHinkleyDetector
from coba.evaluation import (
    EvalResult,
    doubly_robust_eval,
    ncis_eval,
    rejection_sampling_eval,
)
from coba.offpolicy import DoublyRobustUpdater, IPSConfig, IPSEstimator
from coba.router import ClusterRouter
from coba.schemas import BanditDecision, BanditStats, ScoreBreakdown
from coba.types import Arm, PolicyType


# ── Private manager classes extracted from ClusterBandit ─────────────────────


class _DriftManager:
    """Owns per-arm PageHinkley drift detectors and handles reset logic."""

    def __init__(self, enable: bool, arms: list[Arm], delta: float, lambda_: float) -> None:
        self._detectors: dict[Arm, PageHinkleyDetector] | None = None
        self.detected_last_step: bool = False
        if enable:
            self._detectors = {
                arm: PageHinkleyDetector(delta=delta, lambda_=lambda_) for arm in arms
            }

    def add_arm(self, arm: Arm, delta: float, lambda_: float) -> None:
        if self._detectors is not None:
            self._detectors[arm] = PageHinkleyDetector(delta=delta, lambda_=lambda_)

    def remove_arm(self, arm: Arm) -> None:
        if self._detectors is not None:
            self._detectors.pop(arm, None)

    def __contains__(self, arm: Arm) -> bool:
        return self._detectors is not None and arm in self._detectors

    def __getitem__(self, arm: Arm) -> PageHinkleyDetector:
        assert self._detectors is not None
        return self._detectors[arm]

    def check(self, reward: float, arm: Arm, router: ClusterRouter) -> bool:
        """Feed reward to the arm's detector; trigger router reset on alarm."""
        self.detected_last_step = False
        if self._detectors is None or arm not in self._detectors:
            return False
        if self._detectors[arm].update(reward):
            logger.warning("Drift detected for arm {arm} — resetting cluster models", arm=arm)
            router.reset_arm(arm)
            self._detectors[arm].reset()
            self.detected_last_step = True
            return True
        return False


class _ConstraintManager:
    """Owns minimum pull-rate constraints and total decision counter."""

    def __init__(self, min_pull_rates: dict[Arm, float] | None, arms: list[Arm]) -> None:
        if min_pull_rates is not None:
            unknown = set(min_pull_rates) - set(arms)
            if unknown:
                raise ValueError(f"min_pull_rates contains unknown arms: {unknown}")
            for arm, rate in min_pull_rates.items():
                if not (0.0 < rate <= 1.0):
                    raise ValueError(f"min_pull_rates[{arm!r}] must be in (0, 1], got {rate}")
            total = sum(min_pull_rates.values())
            if total > 1.0:
                raise ValueError(f"sum of min_pull_rates must be ≤ 1.0, got {total:.4f}")
        self._rates: dict[Arm, float] | None = dict(min_pull_rates) if min_pull_rates else None
        self._total_decisions: int = 0

    def remove_arm(self, arm: Arm) -> None:
        if self._rates is not None:
            self._rates.pop(arm, None)

    def record_decision(self) -> None:
        self._total_decisions += 1

    def filter_candidates(
        self, all_scores: dict[Arm, float], arm_stats: dict[Arm, BanditStats]
    ) -> dict[Arm, float]:
        """Restrict candidate scores to under-pulled arms when constraints active."""
        if self._rates is None or self._total_decisions == 0:
            return all_scores
        forced: set[Arm] = {
            arm
            for arm, min_rate in self._rates.items()
            if arm_stats[arm].n_pulls / self._total_decisions < min_rate
        }
        return {a: s for a, s in all_scores.items() if a in forced} if forced else all_scores


class ClusterBandit:
    """High-level contextual bandit with KMeans cluster routing.

    Args:
        arms: List of arms, e.g. [1.0, 1.1, 1.2, 1.5].
        n_features: Dimensionality of context vectors.
        policy: Learning policy. Recommended: LIN_UCB or LIN_TS.
        n_clusters: Number of clusters for the KMeans router.
        alpha: Exploration parameter for LinUCB / UCB1.
        v_sq: Posterior variance for LinTS.
        l2_lambda: L2 regularization strength for linear models.
        use_minibatch: Use MiniBatchKMeans for faster online fitting.
        scale_contexts: Normalize context features before clustering.
        seed: Random seed for reproducibility.
        base_estimator: A scikit-learn compatible estimator for non-linear policies.
        n_bootstraps: Number of bootstrapped models for bootstrapped policies.
        epsilon: Epsilon value for epsilon-greedy exploration.
        enable_drift_detection: If True, monitor per-arm reward streams with a
            PageHinkleyDetector. Detected shift resets that arm's cluster models.
        drift_delta: Minimum detectable change magnitude for the PH test.
        drift_lambda: Detection threshold for the PH test (higher → fewer false alarms).
        min_pull_rates: Optional dict mapping arm → minimum fraction of decisions
            that must go to that arm, e.g. {"new_arm": 0.05} guarantees at least
            5% exploration. Arms below their threshold are forced into the candidate
            set. Sum of all rates must be ≤ 1.0.
        n_shared_features: Number of shared context dimensions for
            PolicyType.LIN_UCB_HYBRID. The first n_shared_features elements of
            each context vector are treated as global (user/session) features
            learned jointly across all arms; the remaining n_features - n_shared_features
            elements are arm-specific. Must be 0 for all other policy types.
        neural_embedding_dim: Dimensionality of the penultimate MLP layer for
            PolicyType.NEURAL_LINEAR. This is the size of the embedding fed to LinTS.
        neural_hidden_sizes: Hidden layer widths for the NEURAL_LINEAR backbone MLP.
            An additional output layer of width neural_embedding_dim is appended.
        neural_retrain_freq: Number of total arm updates before retraining the
            NEURAL_LINEAR backbone. Lower → more frequent retrains (slower but
            adapts faster). Higher → faster but backbone lags recent data.
        gp_beta: UCB exploration coefficient for PolicyType.GP_UCB. Higher → more
            exploration (wider confidence intervals).
        gp_length_scale: RBF kernel bandwidth for GP_UCB. Controls how quickly
            reward correlation decays with distance in context space.
        gp_noise_var: Observation noise variance (σ²) for GP_UCB. Acts like L2
            regularisation — higher values reduce overfitting to sparse data.
        gp_max_obs: Maximum number of stored observations for GP_UCB. Oldest
            observations are dropped (FIFO) to keep O(n²) inference tractable.
    """

    def __init__(
        self,
        arms: list[Arm],
        n_features: int,
        config: BanditConfig | None = None,
        # ---- backwards-compatible kwargs (override config when provided) ----
        policy: PolicyType | None = None,
        n_clusters: int | None = None,
        alpha: float | None = None,
        v_sq: float | None = None,
        l2_lambda: float | None = None,
        use_minibatch: bool | None = None,
        scale_contexts: bool | None = None,
        seed: int | None = None,
        base_estimator: "Any | None" = None,
        n_bootstraps: int | None = None,
        epsilon: float | None = None,
        gamma: float | None = None,
        enable_drift_detection: bool | None = None,
        drift_delta: float | None = None,
        drift_lambda: float | None = None,
        min_pull_rates: dict[Arm, float] | None = None,
        n_shared_features: int | None = None,
        neural_embedding_dim: int | None = None,
        neural_hidden_sizes: tuple[int, ...] | None = None,
        neural_retrain_freq: int | None = None,
        gp_beta: float | None = None,
        gp_length_scale: float | None = None,
        gp_noise_var: float | None = None,
        gp_max_obs: int | None = None,
    ) -> None:
        # Build effective config: start from provided config (or defaults),
        # then overlay any explicitly passed kwargs.
        base = config or BanditConfig()
        cfg = BanditConfig(
            policy=policy if policy is not None else base.policy,
            n_clusters=n_clusters if n_clusters is not None else base.n_clusters,
            alpha=alpha if alpha is not None else base.alpha,
            v_sq=v_sq if v_sq is not None else base.v_sq,
            l2_lambda=l2_lambda if l2_lambda is not None else base.l2_lambda,
            gamma=gamma if gamma is not None else base.gamma,
            seed=seed if seed is not None else base.seed,
            use_minibatch=use_minibatch if use_minibatch is not None else base.use_minibatch,
            scale_contexts=scale_contexts if scale_contexts is not None else base.scale_contexts,
            epsilon=epsilon if epsilon is not None else base.epsilon,
            n_bootstraps=n_bootstraps if n_bootstraps is not None else base.n_bootstraps,
            base_estimator=base_estimator if base_estimator is not None else base.base_estimator,
            n_shared_features=(
                n_shared_features if n_shared_features is not None else base.n_shared_features
            ),
            neural_embedding_dim=(
                neural_embedding_dim
                if neural_embedding_dim is not None
                else base.neural_embedding_dim
            ),
            neural_hidden_sizes=(
                neural_hidden_sizes if neural_hidden_sizes is not None else base.neural_hidden_sizes
            ),
            neural_retrain_freq=(
                neural_retrain_freq if neural_retrain_freq is not None else base.neural_retrain_freq
            ),
            gp_beta=gp_beta if gp_beta is not None else base.gp_beta,
            gp_length_scale=(
                gp_length_scale if gp_length_scale is not None else base.gp_length_scale
            ),
            gp_noise_var=gp_noise_var if gp_noise_var is not None else base.gp_noise_var,
            gp_max_obs=gp_max_obs if gp_max_obs is not None else base.gp_max_obs,
            rf_n_estimators=base.rf_n_estimators,
            rf_max_depth=base.rf_max_depth,
            rf_min_samples_leaf=base.rf_min_samples_leaf,
            rf_max_obs=base.rf_max_obs,
            rf_min_uncertainty=base.rf_min_uncertainty,
            enable_drift_detection=(
                enable_drift_detection
                if enable_drift_detection is not None
                else base.enable_drift_detection
            ),
            drift_delta=drift_delta if drift_delta is not None else base.drift_delta,
            drift_lambda=drift_lambda if drift_lambda is not None else base.drift_lambda,
            min_pull_rates=min_pull_rates if min_pull_rates is not None else base.min_pull_rates,
        )
        self._config = cfg

        self.arms: list[Arm] = list(arms)
        self.policy = cfg.policy
        self.n_features = n_features

        self._router = ClusterRouter(
            arms=self.arms,
            n_features=n_features,
            config=cfg,
        )

        # Per-arm statistics for monitoring (pull counts, mean reward)
        self._arm_stats: dict[Arm, BanditStats] = {arm: BanditStats(arm=arm) for arm in self.arms}

        # Extracted concerns
        # Round-robin counter for cold-start arm selection.
        # Ensures all arms are explored before the cluster model is fitted,
        # which is critical for LinUCB (bounded scores → arm A would win forever).
        self._cold_start_counter: int = 0

        self._drift = _DriftManager(
            enable=cfg.enable_drift_detection,
            arms=self.arms,
            delta=cfg.drift_delta,
            lambda_=cfg.drift_lambda,
        )
        self._constraints = _ConstraintManager(
            min_pull_rates=cfg.min_pull_rates,
            arms=self.arms,
        )

    # ------------------------------------------------------------------
    # Core Online API
    # ------------------------------------------------------------------
    def _validate_context_vector(self, context: np.ndarray) -> np.ndarray:
        x = np.asarray(context, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError(f"context must be a 1D feature vector, got shape={x.shape!r}")
        if x.shape[0] != self.n_features:
            raise ValueError(
                f"context feature length mismatch: expected {self.n_features}, got {x.shape[0]}"
            )
        if not np.all(np.isfinite(x)):
            raise ValueError("context contains non-finite values")
        return x

    def _validate_batch_inputs(
        self,
        contexts: np.ndarray,
        arms: np.ndarray,
        rewards: np.ndarray,
        propensities: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ctx = np.asarray(contexts, dtype=np.float64)
        arm_arr = np.asarray(arms)
        rew = np.asarray(rewards, dtype=np.float64)

        if ctx.ndim != 2:
            raise ValueError(f"contexts must be 2D, got shape={ctx.shape!r}")
        if ctx.shape[1] != self.n_features:
            raise ValueError(
                f"contexts feature mismatch: expected {self.n_features}, got {ctx.shape[1]}"
            )
        if not (len(ctx) == len(arm_arr) == len(rew)):
            raise ValueError("contexts, arms, and rewards must have the same length")
        if not np.all(np.isfinite(ctx)) or not np.all(np.isfinite(rew)):
            raise ValueError("contexts/rewards contain non-finite values")

        if propensities is None:
            p = np.ones(len(rew), dtype=np.float64)
        else:
            p = np.asarray(propensities, dtype=np.float64)
            if len(p) != len(rew):
                raise ValueError("propensities must match rewards length")
            if not np.all(np.isfinite(p)):
                raise ValueError("propensities contain non-finite values")
            if np.any(p <= 0):
                raise ValueError("propensities must be > 0")
        return ctx, arm_arr, rew, p

    # ------------------------------------------------------------------
    # decide() private helpers
    # ------------------------------------------------------------------

    def _cold_start_decision(self) -> BanditDecision:
        """Round-robin through arms when the router is not yet fitted.

        Hard-coding arms[0] would starve LinUCB: its bounded score means arm A
        outscores unfitted B, C … permanently.  UCB1 avoids this via inf for
        zero-pull arms, but LinUCB does not.  Round-robin guarantees every arm
        receives at least one sample before the real policy takes over.
        """
        arm = self.arms[self._cold_start_counter % len(self.arms)]
        self._cold_start_counter += 1
        logger.debug(
            "Bandit not fitted yet — cold-start round-robin → {arm} ({idx}/{n})",
            arm=arm,
            idx=self._cold_start_counter,
            n=len(self.arms),
        )
        self._constraints.record_decision()
        all_scores = {str(a): 0.0 for a in self.arms}
        return BanditDecision(
            chosen_arm=arm,
            score=0.0,
            all_scores=all_scores,
            score_breakdown={
                str(a): ScoreBreakdown(score=0.0, mean_estimate=0.0, confidence_width=0.0)
                for a in self.arms
            },
        )

    def _select_arm(
        self,
        candidate_scores: dict[Arm, float],
        all_scores: dict[Arm, float],
        min_confidence_gap: float,
    ) -> BanditDecision:
        """Sort candidates, optionally abstain, and return a BanditDecision."""
        sorted_arms = sorted(candidate_scores.items(), key=lambda kv: kv[1], reverse=True)
        chosen_arm, best_score = sorted_arms[0]
        all_scores_str = {str(k): v for k, v in all_scores.items()}

        if min_confidence_gap > 0.0 and len(sorted_arms) >= 2:
            gap = best_score - sorted_arms[1][1]
            if gap < min_confidence_gap:
                logger.debug(
                    "decide() abstaining: gap={gap:.4f} < threshold={thr:.4f}",
                    gap=gap,
                    thr=min_confidence_gap,
                )
                self._constraints.record_decision()
                return BanditDecision(
                    chosen_arm=None,
                    score=best_score,
                    all_scores=all_scores_str,
                    abstained=True,
                )

        self._constraints.record_decision()
        return BanditDecision(
            chosen_arm=chosen_arm,
            score=best_score,
            all_scores=all_scores_str,
        )

    def _build_score_breakdown(
        self, context: np.ndarray, all_scores: dict[Any, float]
    ) -> dict[str, ScoreBreakdown]:
        """Build per-arm score breakdown for the routed cluster."""
        model_state = self._router.arm_model_state(context)
        arms_state = model_state["arms"]
        breakdown: dict[str, ScoreBreakdown] = {}
        for arm, score in all_scores.items():
            state = arms_state.get(str(arm), {})
            breakdown[str(arm)] = ScoreBreakdown(
                score=float(score),
                mean_estimate=state.get("mean_estimate"),
                confidence_width=state.get("confidence_width"),
                is_fitted=bool(state.get("is_fitted", False)),
                n_obs=state.get("n_obs"),
            )
        return breakdown

    def _decorate_decision(self, context: np.ndarray, decision: BanditDecision) -> BanditDecision:
        """Populate observability fields on a selected decision."""
        if decision.chosen_arm is None:
            return decision

        mean_est, ucb_width = self._router.score_decomposed_for_arm(context, decision.chosen_arm)
        updates: dict = {
            "mean_estimate": mean_est,
            "confidence_width": ucb_width,
            "score_breakdown": self._build_score_breakdown(context, decision.all_scores),
        }

        if self._config.policy == PolicyType.EPSILON_GREEDY:
            cluster_idx, _scaled_ctx = self._router._route(context)
            model = self._router._cluster_bandits[cluster_idx].get(decision.chosen_arm)
            if model is not None and hasattr(model, "last_was_random"):
                updates["was_random"] = model.last_was_random

        return decision.model_copy(update=updates)

    def decide(
        self,
        context: np.ndarray,
        min_confidence_gap: float = 0.0,
    ) -> BanditDecision:
        """Select the optimal arm for the given context vector.

        If the bandit is not yet fitted (cold start), falls back to the first
        arm in the list to avoid errors.

        Args:
            context: Feature vector, shape (n_features,).
            min_confidence_gap: If > 0, abstain when the score gap between the
                best and second-best arm is smaller than this threshold.
                Returns BanditDecision(chosen_arm=None, abstained=True) so the
                caller can apply a fallback (e.g. a rule-based policy).
                Default 0.0 disables abstention.

        Returns:
            BanditDecision with the chosen arm and score breakdown.
            Check `.abstained` before using `.chosen_arm`.
        """
        x = self._validate_context_vector(context)
        if not self._router.is_fitted:
            return self._cold_start_decision()
        all_scores = self._router.score_all(x)
        candidate_scores = self._constraints.filter_candidates(all_scores, self._arm_stats)
        decision = self._select_arm(candidate_scores, all_scores, min_confidence_gap)
        return self._decorate_decision(x, decision)

    def decide_top_k(self, context: np.ndarray, k: int) -> list[tuple[Arm, float]]:
        """Return the top-k arms ranked by score for the given context.

        Useful for ranked recommendation, list display, or exploration of
        the second-best option when the best arm is unavailable.

        Args:
            context: Feature vector, shape (n_features,).
            k: Number of arms to return. Clamped to len(arms) if k > n_arms.

        Returns:
            List of (arm, score) tuples in descending score order, length min(k, n_arms).
        """
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")

        x = self._validate_context_vector(context)
        k = min(k, len(self.arms))

        if not self._router.is_fitted:
            logger.debug("Bandit not fitted yet — returning first {k} arms for cold start", k=k)
            return [(arm, 0.0) for arm in self.arms[:k]]

        all_scores = self._router.score_all(x)
        return sorted(all_scores.items(), key=lambda kv: kv[1], reverse=True)[:k]

    def score_all(self, context: np.ndarray) -> dict[Arm, float]:
        """Return per-arm scores for a single context."""
        x = self._validate_context_vector(context)
        return self._router.score_all(x)

    def decide_batch(
        self,
        contexts: np.ndarray,
        min_confidence_gap: float = 0.0,
    ) -> list[BanditDecision]:
        """Vectorized batch decision for multiple context vectors.

        Scores all contexts in one shot using vectorized KMeans assignment,
        then dispatches per-cluster arm scoring.  Significantly faster than
        calling ``decide()`` in a Python loop when N is large (e.g. ≥100).

        Args:
            contexts: Feature matrix, shape (n_samples, n_features).
            min_confidence_gap: Applied per-row; same semantics as ``decide()``.

        Returns:
            List of BanditDecision objects, one per row of ``contexts``.
        """
        contexts_arr = np.asarray(contexts, dtype=np.float64)
        if contexts_arr.ndim != 2:
            raise ValueError(f"contexts must be 2D, got shape={contexts_arr.shape!r}")
        if contexts_arr.shape[1] != self.n_features:
            raise ValueError(
                f"contexts feature mismatch: expected {self.n_features}, got {contexts_arr.shape[1]}"
            )
        if not self._router.is_fitted:
            # Cold start: mirror decide() and round-robin all rows.
            return [self._cold_start_decision() for _ in range(len(contexts_arr))]

        decisions: list[BanditDecision] = []
        for x in contexts_arr:
            all_scores = self._router.score_all(x)
            candidate_scores = self._constraints.filter_candidates(all_scores, self._arm_stats)
            decision = self._select_arm(candidate_scores, all_scores, min_confidence_gap)
            decisions.append(self._decorate_decision(x, decision))
        return decisions

    def _update_arm_stats(self, arm: Arm, reward: float) -> None:
        """Delegate to BanditStats.record() — Welford update lives there."""
        if arm in self._arm_stats:
            self._arm_stats[arm].record(reward)

    def update(
        self,
        context: np.ndarray,
        arm: Arm,
        reward: float,
        propensity: float = 1.0,
    ) -> None:
        """Update the bandit model with the observed reward for a past decision.

        Applies IPS weight correction using the propensity.

        Args:
            context: Feature vector, shape (n_features,).
            arm: The arm that was actually applied.
            reward: Observed scalar reward.
            propensity: Probability of the logging policy choosing this arm.
                        Default 1.0 disables IPS correction.
        """
        x = self._validate_context_vector(context)
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        if not np.isfinite(propensity) or propensity <= 0:
            raise ValueError("propensity must be a finite value > 0")

        # Delegate IPS clipping to IPSEstimator — single source of truth.
        weight = float(IPSEstimator.compute_weights(np.array([propensity]))[0])

        self._router.update(x, arm, reward, weight)
        self._update_arm_stats(arm, reward)
        self._drift.check(reward, arm, self._router)

    def update_batch(
        self,
        contexts: np.ndarray,
        arms: np.ndarray,
        rewards: np.ndarray,
        propensities: np.ndarray | None = None,
    ) -> None:
        """Batch update from parallel arrays.

        Args:
            contexts: Feature matrix, shape (n_samples, n_features).
            arms: Array of chosen arms, shape (n_samples,).
            rewards: Array of observed rewards, shape (n_samples,).
            propensities: Array of propensities. None implies uniform 1.0.
        """
        if len(rewards) == 0:
            return

        contexts, arms, rewards, p = self._validate_batch_inputs(
            contexts=contexts,
            arms=arms,
            rewards=rewards,
            propensities=propensities,
        )
        weights = np.clip(1.0 / np.maximum(p, 1e-4), 0.0, 10.0)

        self._router.partial_fit(contexts, arms, rewards, weights)

        # Update monitoring stats
        for arm_val, reward_val in zip(arms, rewards):
            self._update_arm_stats(arm_val, float(reward_val))

        logger.info("Batch updated {n} observations", n=len(rewards))

    # ------------------------------------------------------------------
    # Off-policy Bootstrap
    # ------------------------------------------------------------------

    def fit_offline(
        self,
        contexts: np.ndarray,
        decisions: np.ndarray,
        rewards: np.ndarray,
        propensities: np.ndarray | None = None,
        use_dr: bool = False,
        reward_estimates: np.ndarray | None = None,
    ) -> "ClusterBandit":
        """Bootstrap the bandit from historical logs with IPS/DR correction.

        Args:
            contexts: Feature matrix, shape (n_samples, n_features).
            decisions: Logged arm decisions, shape (n_samples,).
            rewards: Logged rewards, shape (n_samples,).
            propensities: Propensity of the logging policy choosing each arm.
            use_dr: If True, apply Doubly-Robust correction (requires reward_estimates).
            reward_estimates: Direct reward model estimates for DR. Shape (n_samples,).

        Returns:
            self (for method chaining)
        """
        if propensities is None:
            n_arms = len(self.arms)
            propensities = np.full(len(rewards), 1.0 / n_arms, dtype=np.float64)
            logger.info(
                "No propensities provided — assuming uniform logging policy (p=1/{n})",
                n=n_arms,
            )

        contexts_arr, decisions_arr, rewards_arr, prop_arr = self._validate_batch_inputs(
            contexts=contexts,
            arms=decisions,
            rewards=rewards,
            propensities=propensities,
        )
        if use_dr and reward_estimates is None:
            raise ValueError("reward_estimates must be provided when use_dr=True")
        if reward_estimates is not None:
            reward_estimates = np.asarray(reward_estimates, dtype=np.float64)
            if len(reward_estimates) != len(rewards_arr):
                raise ValueError("reward_estimates must match rewards length")

        ips_config = IPSConfig(clip_min=1e-4, clip_max=10.0, use_dr=use_dr)
        updater = DoublyRobustUpdater(self._router, config=ips_config)
        updater.fit_offline(
            contexts=contexts_arr,
            decisions=decisions_arr,
            rewards=rewards_arr,
            propensities=prop_arr,
            reward_estimates=reward_estimates,
        )
        return self

    # ------------------------------------------------------------------
    # Arm Management
    # ------------------------------------------------------------------

    def add_arm(
        self,
        arm: Arm,
        warm_start_from: Arm | None = None,
        gamma: float | None = None,
    ) -> None:
        """Dynamically add a new arm to the bandit.

        Args:
            arm: New arm identifier.
            warm_start_from: If given, initialize the new arm's model by copying
                             this arm's trained model.
            gamma: Per-arm discount factor. Overrides the bandit-level gamma for
                   this arm only. A higher decay (e.g. 0.9) lets a newly launched
                   arm adapt faster to its own reward distribution without affecting
                   existing arms.
        """
        self._router.add_arm(arm, warm_start_from=warm_start_from, gamma=gamma)
        self.arms = self._router.arms
        self._arm_stats[arm] = BanditStats(arm=arm)
        self._drift.add_arm(arm, delta=self._config.drift_delta, lambda_=self._config.drift_lambda)
        logger.info(
            "Added arm {arm} (warm_start_from={src})",
            arm=arm,
            src=warm_start_from,
        )

    def remove_arm(self, arm: Arm) -> None:
        """Remove an arm from the bandit.

        Args:
            arm: Arm identifier to remove.
        """
        self._router.remove_arm(arm)
        self.arms = self._router.arms
        self._arm_stats.pop(arm, None)
        self._drift.remove_arm(arm)
        self._constraints.remove_arm(arm)
        logger.info("Removed arm {arm}", arm=arm)

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def get_stats(self) -> list[BanditStats]:
        """Return per-arm statistics for monitoring dashboards."""
        return list(self._arm_stats.values())

    def get_model_state(self, context: np.ndarray) -> dict:
        """Return routed-cluster per-arm model internals for debugging dashboards.

        This intentionally exposes a stable, JSON-friendly subset of internal
        state so downstream tools can inspect coba directly instead of keeping
        divergent shadow copies.
        """
        x = self._validate_context_vector(context)
        state = self._router.arm_model_state(x)
        return {
            "policy": self.policy.value,
            "is_fitted": self.is_fitted,
            "cluster": state["cluster"],
            "arms": state["arms"],
        }

    def evaluate_rejection_sampling(
        self, contexts: np.ndarray, decisions: np.ndarray, rewards: np.ndarray
    ) -> EvalResult:
        """Offline policy estimate using rejection sampling."""
        return rejection_sampling_eval(self._router, contexts, decisions, rewards)

    def evaluate_doubly_robust(
        self,
        contexts: np.ndarray,
        decisions: np.ndarray,
        rewards: np.ndarray,
        propensities: np.ndarray,
        reward_estimates: np.ndarray,
        target_reward_estimates: np.ndarray | None = None,
        clip_min: float = 1e-4,
        clip_max: float | None = 10.0,
    ) -> EvalResult:
        """Offline policy estimate using doubly-robust correction."""
        return doubly_robust_eval(
            self._router,
            contexts,
            decisions,
            rewards,
            propensities,
            reward_estimates,
            target_reward_estimates=target_reward_estimates,
            clip_min=clip_min,
            clip_max=clip_max,
        )

    def evaluate_ncis(
        self,
        policy_scores: np.ndarray,
        logging_scores: np.ndarray,
        rewards: np.ndarray,
        clip_min: float = 1e-8,
        clip_max: float = 1e3,
    ) -> EvalResult:
        """Offline policy estimate using normalized capped importance sampling."""
        return ncis_eval(
            policy_scores=policy_scores,
            logging_scores=logging_scores,
            rewards=rewards,
            clip_min=clip_min,
            clip_max=clip_max,
        )

    def get_cluster_assignment(self, context: np.ndarray) -> int:
        """Return which cluster this context belongs to.

        Args:
            context: Feature vector.
        Returns:
            Cluster index (0 to n_clusters-1).
        """
        x = self._validate_context_vector(context)
        cluster_idx, _ = self._router._route(x)
        return cluster_idx

    @property
    def is_fitted(self) -> bool:
        """True if the bandit has been trained on data."""
        return self._router.is_fitted

    @property
    def n_clusters(self) -> int:
        return self._router.n_clusters
