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

from coba.drift import PageHinkleyDetector
from coba.evaluation import (
    EvalResult,
    doubly_robust_eval,
    ncis_eval,
    rejection_sampling_eval,
)
from coba.offpolicy import DoublyRobustUpdater, IPSConfig
from coba.router import ClusterRouter
from coba.schemas import BanditDecision, BanditStats
from coba.types import Arm, PolicyType


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
        policy: PolicyType = PolicyType.LIN_UCB,
        n_clusters: int = 5,
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
        enable_drift_detection: bool = False,
        drift_delta: float = 0.005,
        drift_lambda: float = 50.0,
        min_pull_rates: dict[Arm, float] | None = None,
        n_shared_features: int = 0,
        neural_embedding_dim: int = 16,
        neural_hidden_sizes: tuple[int, ...] = (64, 32),
        neural_retrain_freq: int = 200,
        gp_beta: float = 2.0,
        gp_length_scale: float = 1.0,
        gp_noise_var: float = 0.1,
        gp_max_obs: int = 500,
    ) -> None:
        self.arms: list[Arm] = list(arms)
        self.policy = policy
        self.n_features = n_features

        self._router = ClusterRouter(
            arms=self.arms,
            n_clusters=n_clusters,
            policy=policy,
            n_features=n_features,
            alpha=alpha,
            v_sq=v_sq,
            l2_lambda=l2_lambda,
            use_minibatch=use_minibatch,
            scale_contexts=scale_contexts,
            seed=seed,
            base_estimator=base_estimator,
            n_bootstraps=n_bootstraps,
            epsilon=epsilon,
            gamma=gamma,
            n_shared_features=n_shared_features,
            neural_embedding_dim=neural_embedding_dim,
            neural_hidden_sizes=neural_hidden_sizes,
            neural_retrain_freq=neural_retrain_freq,
            gp_beta=gp_beta,
            gp_length_scale=gp_length_scale,
            gp_noise_var=gp_noise_var,
            gp_max_obs=gp_max_obs,
        )

        # Per-arm statistics for monitoring (pull counts, mean reward)
        self._arm_stats: dict[Arm, BanditStats] = {arm: BanditStats(arm=arm) for arm in self.arms}

        # Optional per-arm drift detectors
        self._drift_detectors: dict[Arm, PageHinkleyDetector] | None = None
        if enable_drift_detection:
            self._drift_detectors = {
                arm: PageHinkleyDetector(delta=drift_delta, lambda_=drift_lambda)
                for arm in self.arms
            }
            self._drift_delta = drift_delta
            self._drift_lambda = drift_lambda

        # Optional minimum pull-rate constraints {arm: fraction in (0, 1]}
        if min_pull_rates is not None:
            unknown = set(min_pull_rates) - set(self.arms)
            if unknown:
                raise ValueError(f"min_pull_rates contains unknown arms: {unknown}")
            for arm, rate in min_pull_rates.items():
                if not (0.0 < rate <= 1.0):
                    raise ValueError(f"min_pull_rates[{arm!r}] must be in (0, 1], got {rate}")
            total = sum(min_pull_rates.values())
            if total > 1.0:
                raise ValueError(f"sum of min_pull_rates must be ≤ 1.0, got {total:.4f}")
        self._min_pull_rates: dict[Arm, float] | None = min_pull_rates
        self._total_decisions: int = 0

    # ------------------------------------------------------------------
    # Core Online API
    # ------------------------------------------------------------------
    def _validate_context_vector(self, context: np.ndarray) -> np.ndarray:
        x = np.asarray(context, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError(f"context must be a 1D feature vector, got shape={x.shape!r}")
        if x.shape[0] != self.n_features:
            raise ValueError(
                f"context feature length mismatch: expected {self.n_features}, " f"got {x.shape[0]}"
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
                f"contexts feature mismatch: expected {self.n_features}, " f"got {ctx.shape[1]}"
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
            logger.debug("Bandit not fitted yet — returning base arm for cold start")
            base_arm = self.arms[0]
            self._total_decisions += 1
            return BanditDecision(
                chosen_arm=base_arm,
                score=0.0,
                all_scores={str(a): 0.0 for a in self.arms},
            )

        all_scores = self._router.score_all(x)

        # Minimum pull-rate enforcement: force under-pulled arms into the candidate set
        if self._min_pull_rates is not None and self._total_decisions > 0:
            forced: set[Arm] = set()
            for arm, min_rate in self._min_pull_rates.items():
                arm_pulls = self._arm_stats[arm].n_pulls
                actual_rate = arm_pulls / self._total_decisions
                if actual_rate < min_rate:
                    forced.add(arm)
            if forced:
                # Restrict scores to forced arms only; pick best among them
                candidate_scores = {a: s for a, s in all_scores.items() if a in forced}
            else:
                candidate_scores = all_scores
        else:
            candidate_scores = all_scores

        sorted_arms = sorted(candidate_scores.items(), key=lambda kv: kv[1], reverse=True)
        chosen_arm, best_score = sorted_arms[0]

        if min_confidence_gap > 0.0 and len(sorted_arms) >= 2:
            second_score = sorted_arms[1][1]
            gap = best_score - second_score
            if gap < min_confidence_gap:
                logger.debug(
                    "decide() abstaining: gap={gap:.4f} < threshold={thr:.4f}",
                    gap=gap,
                    thr=min_confidence_gap,
                )
                self._total_decisions += 1
                return BanditDecision(
                    chosen_arm=None,
                    score=best_score,
                    all_scores={str(k): v for k, v in all_scores.items()},
                    abstained=True,
                )

        self._total_decisions += 1
        return BanditDecision(
            chosen_arm=chosen_arm,
            score=best_score,
            all_scores={str(k): v for k, v in all_scores.items()},
        )

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

    def _update_arm_stats(self, arm: Arm, reward: float) -> None:
        """Increment pull count and update running mean reward for an arm.

        Uses Welford's online algorithm to avoid float accumulation drift.
        """
        if arm not in self._arm_stats:
            return
        stats = self._arm_stats[arm]
        stats.n_pulls += 1
        # Welford's running mean: mean += (x - mean) / n
        stats.mean_reward += (reward - stats.mean_reward) / stats.n_pulls

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

        # IPS weight: 1/propensity, capped to avoid explosion from near-zero propensities
        weight = min(1.0 / max(propensity, 1e-4), 10.0)

        self._router.update(x, arm, reward, weight)
        self._update_arm_stats(arm, reward)

        if self._drift_detectors is not None and arm in self._drift_detectors:
            if self._drift_detectors[arm].update(reward):
                logger.warning(
                    "Drift detected for arm {arm} — resetting cluster models",
                    arm=arm,
                )
                self._router.reset_arm(arm)
                self._drift_detectors[arm].reset()

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
        if self._drift_detectors is not None:
            self._drift_detectors[arm] = PageHinkleyDetector(
                delta=self._drift_delta, lambda_=self._drift_lambda
            )
        # New arms are not automatically constrained; caller sets min_pull_rates at init
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
        if self._drift_detectors is not None:
            self._drift_detectors.pop(arm, None)
        if self._min_pull_rates is not None:
            self._min_pull_rates.pop(arm, None)
        logger.info("Removed arm {arm}", arm=arm)

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def get_stats(self) -> list[BanditStats]:
        """Return per-arm statistics for monitoring dashboards."""
        return list(self._arm_stats.values())

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
