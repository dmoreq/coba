"""
Offline evaluation metrics for contextual bandit policies.

Provides three evaluation methodologies from the literature:

1. **Rejection Sampling** (Li et al., 2010):
   Unbiased if logs were collected uniformly at random. Simulates an online
   bandit by accepting only the samples where the bandit's choice matches the
   logged arm. Fast and simple, but wastes many samples.

2. **Doubly-Robust Policy Evaluation** (Dudík et al., 2011):
   Combines a reward model (direct method) with IPS correction for lower
   variance than pure IPS and lower bias than pure direct method.
   Preferred when propensities are reliable.

3. **Normalized Capped Importance Sampling (NCIS)** (Gilotte et al., 2018):
   IPS with weight normalization and a cap — reduces extreme weight variance
   at the cost of introducing slight bias. Useful when propensities are small.
"""

from dataclasses import dataclass

import numpy as np
from loguru import logger

from coba.routers.cluster_router import ClusterRouter


@dataclass
class EvalResult:
    """Result from an offline evaluation run."""

    method: str
    estimated_reward: float
    n_samples_used: int
    n_samples_total: int

    @property
    def utilization_rate(self) -> float:
        """Fraction of log samples actually used in evaluation."""
        if self.n_samples_total == 0:
            return 0.0
        return self.n_samples_used / self.n_samples_total

    def __repr__(self) -> str:
        return (
            f"EvalResult(method={self.method!r}, "
            f"estimated_reward={self.estimated_reward:.4f}, "
            f"n_used={self.n_samples_used}/{self.n_samples_total} "
            f"[{self.utilization_rate:.1%}])"
        )


def rejection_sampling_eval(
    router: ClusterRouter,
    contexts: np.ndarray,
    decisions: np.ndarray,
    rewards: np.ndarray,
) -> EvalResult:
    """Evaluate a bandit policy using rejection sampling.

    For each sample in the log, ask the bandit what arm it would choose.
    Accept the sample only if the bandit's choice matches the logged arm.
    The estimated reward is the mean reward over accepted samples.

    Note:
        Unbiased only if logs were collected by a uniform random policy.
        For non-uniform logging policies, use doubly_robust_eval instead.

    Args:
        router: Fitted ClusterRouter to evaluate.
        contexts: Feature matrix, shape (n_samples, n_features).
        decisions: Logged arm decisions, shape (n_samples,).
        rewards: Logged rewards, shape (n_samples,).

    Returns:
        EvalResult with estimated reward and acceptance statistics.
    """
    if not router.is_fitted:
        raise ValueError("ClusterRouter must be fitted before evaluation.")

    contexts = np.asarray(contexts, dtype=np.float64)
    decisions = np.asarray(decisions)
    rewards = np.asarray(rewards, dtype=np.float64)

    n_total = len(rewards)
    predictions = np.array([router.predict(ctx) for ctx in contexts])
    mask = predictions == decisions
    accepted_rewards = rewards[mask].tolist()

    n_accepted = len(accepted_rewards)
    if n_accepted == 0:
        logger.warning(
            "Rejection sampling: no samples accepted — policy may not overlap with logs"
        )
        return EvalResult("rejection_sampling", 0.0, 0, n_total)

    estimated_reward = float(np.mean(accepted_rewards))
    logger.info(
        "Rejection sampling: accepted {n}/{total}, estimated_reward={r:.4f}",
        n=n_accepted,
        total=n_total,
        r=estimated_reward,
    )
    return EvalResult(
        "rejection_sampling", estimated_reward, n_accepted, n_total
    )


def doubly_robust_eval(
    router: ClusterRouter,
    contexts: np.ndarray,
    decisions: np.ndarray,
    rewards: np.ndarray,
    propensities: np.ndarray,
    reward_estimates: np.ndarray,
    target_reward_estimates: np.ndarray | None = None,
    clip_min: float = 1e-4,
    clip_max: float | None = 10.0,
) -> EvalResult:
    """Doubly-Robust policy evaluation.

    Estimates the expected reward of the bandit policy using the DR estimator:
      DR(i) = reward_model(x_i, pi(x_i)) +
              (r_i - reward_model(x_i, a_i)) / propensity_i   [for a_i == pi(x_i)]
              reward_model(x_i, pi(x_i))                       [for a_i != pi(x_i)]

    Lower variance than pure IPS; lower bias than pure direct method.

    Args:
        router: Fitted ClusterRouter to evaluate.
        contexts: Feature matrix, shape (n_samples, n_features).
        decisions: Logged arm decisions, shape (n_samples,).
        rewards: Logged rewards, shape (n_samples,).
        propensities: Logging policy propensities, shape (n_samples,).
        reward_estimates: Direct method reward model estimates for the LOGGED arm,
                         shape (n_samples,).
        target_reward_estimates: Direct method estimates for the TARGET policy action
                         pi(x), shape (n_samples,). If None, falls back to
                         reward_estimates with a warning.
        clip_min: Minimum propensity clip to avoid explosion.
        clip_max: Maximum weight clip for variance control.

    Returns:
        EvalResult with estimated reward.
    """
    if not router.is_fitted:
        raise ValueError("ClusterRouter must be fitted before evaluation.")

    contexts = np.asarray(contexts, dtype=np.float64)
    decisions = np.asarray(decisions)
    rewards = np.asarray(rewards, dtype=np.float64)
    propensities = np.clip(
        np.asarray(propensities, dtype=np.float64), clip_min, 1.0
    )
    reward_estimates = np.asarray(reward_estimates, dtype=np.float64)
    if target_reward_estimates is None:
        logger.warning(
            "DR eval called without target_reward_estimates; falling back to "
            "reward_estimates for both logged and target actions"
        )
        target_reward_estimates = reward_estimates
    target_reward_estimates = np.asarray(
        target_reward_estimates, dtype=np.float64
    )

    n_total = len(rewards)
    if (
        len(decisions) != n_total
        or len(propensities) != n_total
        or len(reward_estimates) != n_total
        or len(target_reward_estimates) != n_total
    ):
        raise ValueError("All DR inputs must have the same length")

    predictions = np.array([router.predict(ctx) for ctx in contexts])
    action_match = predictions == decisions

    ips_correction = (rewards - reward_estimates) / propensities
    if clip_max is not None:
        ips_correction = np.clip(ips_correction, -clip_max, clip_max)

    dr_values = target_reward_estimates + np.where(
        action_match, ips_correction, 0.0
    )

    estimated_reward = float(np.mean(dr_values))
    logger.info(
        "DR eval: {n} samples, estimated_reward={r:.4f}",
        n=n_total,
        r=estimated_reward,
    )
    return EvalResult("doubly_robust", estimated_reward, n_total, n_total)


def ncis_eval(
    policy_scores: np.ndarray,
    logging_scores: np.ndarray,
    rewards: np.ndarray,
    clip_min: float = 1e-8,
    clip_max: float = 1e3,
) -> EvalResult:
    """Normalized Capped Importance Sampling (NCIS) evaluation.

    Uses the ratio of policy scores to logging scores as importance weights.
    Caps extreme ratios and normalizes to reduce variance.

    Args:
        policy_scores: Score of the evaluated policy for logged actions,
                       shape (n_samples,).
        logging_scores: Score of the logging policy for logged actions,
                        shape (n_samples,).
        rewards: Logged rewards, shape (n_samples,).
        clip_min: Minimum ratio (clips below this value).
        clip_max: Maximum ratio (discards above this value).

    Returns:
        EvalResult with estimated reward.
    """
    policy_scores = np.asarray(policy_scores, dtype=np.float64)
    logging_scores = np.asarray(logging_scores, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)

    # Importance weights: ratio of policy to logging scores
    w = np.clip(
        policy_scores / np.maximum(logging_scores, clip_min),
        a_min=clip_min,
        a_max=None,
    )

    # Discard samples with extreme weights (NCIS cap)
    keep_mask = w <= clip_max
    n_kept = int(np.sum(keep_mask))

    if n_kept == 0:
        raise ValueError(
            "NCIS: no samples below clip_max — adjust clip_max parameter."
        )

    w_kept = w[keep_mask]
    r_kept = rewards[keep_mask]

    # Normalized weighted average
    estimated_reward = float(np.dot(w_kept, r_kept) / np.sum(w_kept))

    logger.info(
        "NCIS eval: kept {n}/{total} samples, estimated_reward={r:.4f}",
        n=n_kept,
        total=len(rewards),
        r=estimated_reward,
    )
    return EvalResult("ncis", estimated_reward, n_kept, len(rewards))
