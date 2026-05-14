"""
Inverse Propensity Scoring (IPS) and Doubly-Robust (DR) off-policy learning.

Background:
  When bootstrapping from historical logs collected by a different policy
  (e.g., a rule-based or random logging policy), the data is biased: certain arms
  were chosen more often than others, and rewards are only observed for the chosen arm.

  IPS corrects for this bias by reweighting each observation by the inverse of
  the probability that the logging policy chose that arm:

    IPS weight = reward / propensity
    IPS weight = min(clip_max, reward / max(propensity, clip_min))  [clipped]

  Doubly-Robust (DR) combines IPS with a reward model (direct method) to reduce
  variance while maintaining unbiasedness:

    DR corrected reward = reward_model(x, a) +
                         (reward - reward_model(x, a)) / propensity

  This gives us two chances to be correct: if the reward model is accurate,
  variance is low even with small propensities; if the model is wrong, the
  IPS term corrects it.

Reference:
  Dudík et al., "Doubly Robust Policy Evaluation and Learning", ICML 2011.
  Gilotte et al., "Offline A/B Testing for Recommender Systems", WSDM 2018.

Usage:
  log_df has columns: context_features, chosen_arm, reward, propensity

  # Compute IPS-corrected rewards
  corrected = IPSEstimator.compute_weights(rewards, propensities)

  # Update a ClusterRouter from historical logs using DR
  dr = DoublyRobustUpdater(router)
  dr.fit_offline(contexts, decisions, rewards, propensities)
"""

from dataclasses import dataclass

import numpy as np
from loguru import logger

from coba.router import ClusterRouter


@dataclass
class IPSConfig:
    """Configuration for IPS / DR off-policy correction.

    Attributes:
        clip_min: Minimum propensity value to avoid division explosion.
                  Observations with propensity < clip_min are clipped.
        clip_max: Maximum IPS weight (caps reward correction to prevent high variance).
                  Set to None to disable max clipping.
        use_dr: If True, applies Doubly-Robust correction (requires a reward model).
    """

    clip_min: float = 1e-4
    clip_max: float | None = 10.0
    use_dr: bool = False
    allow_ips_fallback_when_dr_missing: bool = False


class IPSEstimator:
    """Computes IPS-corrected importance weights for off-policy learning.

    Static utility methods — no state is maintained.
    """

    @staticmethod
    def compute_weights(
        propensities: np.ndarray,
        config: IPSConfig | None = None,
    ) -> np.ndarray:
        """Compute clipped IPS importance weights (1 / propensity).

        Args:
            propensities: Logging policy propensities, shape (n_samples,).
                         Must be in (0, 1].
            config: IPS configuration (clip_min, clip_max).

        Returns:
            IPS weights, shape (n_samples,). These are the `weight` values to
            pass to arm_model.update() for off-policy correction.
        """
        cfg = config or IPSConfig()
        p = np.asarray(propensities, dtype=np.float64)
        p_clipped = np.clip(p, cfg.clip_min, 1.0)

        weights = 1.0 / p_clipped  # raw IPS weights

        if cfg.clip_max is not None:
            n_clipped = int(np.sum(weights > cfg.clip_max))
            if n_clipped > 0:
                logger.debug(
                    "IPS: clipping {n} observations with weight > {max}",
                    n=n_clipped,
                    max=cfg.clip_max,
                )
            weights = np.minimum(weights, cfg.clip_max)

        return weights

    @staticmethod
    def compute_dr_rewards(
        rewards: np.ndarray,
        propensities: np.ndarray,
        reward_estimates: np.ndarray,
        config: IPSConfig | None = None,
    ) -> np.ndarray:
        """Compute Doubly-Robust corrected rewards.

        DR corrected = reward_model_estimate
                       + (reward - reward_model_estimate) / propensity

        Args:
            rewards: Observed rewards, shape (n_samples,).
            propensities: Logging policy propensities, shape (n_samples,).
            reward_estimates: Direct method reward model estimates,
                              shape (n_samples,). These are predictions from
                              an already-fitted model.
            config: IPS configuration.

        Returns:
            DR-corrected rewards, shape (n_samples,).
        """
        cfg = config or IPSConfig()
        p_clipped = np.clip(np.asarray(propensities, dtype=np.float64), cfg.clip_min, 1.0)
        r = np.asarray(rewards, dtype=np.float64)
        rhat = np.asarray(reward_estimates, dtype=np.float64)

        # DR formula: direct estimate + IPS correction on the residual
        dr_rewards = rhat + (r - rhat) / p_clipped

        if cfg.clip_max is not None:
            # Clip extreme DR rewards for variance control
            dr_rewards = np.clip(dr_rewards, -cfg.clip_max, cfg.clip_max)

        return dr_rewards


def _apply_correction(
    config: IPSConfig,
    rewards: np.ndarray,
    propensities: np.ndarray,
    reward_estimates: "np.ndarray | None",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (corrected_rewards, weights) applying DR or IPS depending on config.

    Extracted to avoid duplicating the branching logic between fit_offline and
    update_from_logs.
    """
    if config.use_dr and reward_estimates is not None:
        corrected_rewards = IPSEstimator.compute_dr_rewards(
            rewards, propensities, reward_estimates, config
        )
        return corrected_rewards, np.ones(len(rewards), dtype=np.float64)
    return np.asarray(rewards, dtype=np.float64), IPSEstimator.compute_weights(propensities, config)


class DoublyRobustUpdater:
    """Updates a ClusterRouter from historical off-policy logs using DR/IPS correction.

    This is the primary tool for bootstrapping the bandit from existing historical logs.
    After calling `fit_offline`, the ClusterRouter's arm models will have absorbed
    the historical signal with appropriate propensity correction.

    Args:
        router: The ClusterRouter to update.
        config: IPS/DR configuration.
    """

    def __init__(self, router: ClusterRouter, config: IPSConfig | None = None) -> None:
        self.router = router
        self.config = config or IPSConfig()

    def fit_offline(
        self,
        contexts: np.ndarray,
        decisions: np.ndarray,
        rewards: np.ndarray,
        propensities: np.ndarray,
        reward_estimates: np.ndarray | None = None,
    ) -> None:
        """Fit the ClusterRouter from off-policy historical logs.

        Args:
            contexts: Context feature matrix, shape (n_samples, n_features).
            decisions: Chosen arms per sample, shape (n_samples,).
            rewards: Observed rewards, shape (n_samples,).
            propensities: Logging policy propensities, shape (n_samples,).
            reward_estimates: Direct method reward estimates, shape (n_samples,).
                             Required when config.use_dr=True. Pass None for pure IPS.
        """
        if self.config.use_dr and reward_estimates is None:
            raise ValueError(
                "reward_estimates must be provided when use_dr=True. "
                "Fit a reward model first and pass its predictions."
            )

        corrected_rewards, weights = _apply_correction(
            self.config, rewards, propensities, reward_estimates
        )
        logger.info(
            "Off-policy fit: {n} samples, mean_weight={mw:.3f}, mean_reward={mr:.3f}",
            n=len(rewards),
            mw=float(np.mean(weights)),
            mr=float(np.mean(corrected_rewards)),
        )
        self.router.fit(
            contexts=contexts,
            decisions=decisions,
            rewards=corrected_rewards,
            weights=weights,
        )

    def update_from_logs(
        self,
        contexts: np.ndarray,
        decisions: np.ndarray,
        rewards: np.ndarray,
        propensities: np.ndarray,
        reward_estimates: np.ndarray | None = None,
    ) -> None:
        """Incrementally update the ClusterRouter from a new batch of off-policy logs.

        Use this for streaming/daily log ingestion without refitting the clusters.

        Args:
            contexts: Context matrix, shape (n_samples, n_features).
            decisions: Chosen arms, shape (n_samples,).
            rewards: Observed rewards, shape (n_samples,).
            propensities: Logging policy propensities, shape (n_samples,).
            reward_estimates: DR reward model estimates (optional).
        """
        if (
            self.config.use_dr
            and reward_estimates is None
            and not self.config.allow_ips_fallback_when_dr_missing
        ):
            raise ValueError(
                "reward_estimates must be provided when use_dr=True. "
                "Set allow_ips_fallback_when_dr_missing=True to explicitly fallback to IPS."
            )

        corrected_rewards, weights = _apply_correction(
            self.config, rewards, propensities, reward_estimates
        )
        self.router.partial_fit(
            contexts=contexts,
            decisions=decisions,
            rewards=corrected_rewards,
            weights=weights,
        )
