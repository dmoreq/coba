"""Reward functions for contextual bandit simulation environments.

These are pure mathematical functions with no dependency on any web
framework. They can be used by the coba-web Dash application, standalone
simulation scripts, or research notebooks.

All discrete reward functions return a Bernoulli sample (0 or 1).
All continuous reward functions return a float in [0, 1].
"""

from collections.abc import Callable

import numpy as np

# Type alias: (arm_name, context_vector) → reward
RewardFn = Callable[[str, np.ndarray], float]


def _sigmoid(x: float) -> float:
    """Sigmoid function with clipping for numerical stability."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def categorical_reward(rates: dict[str, float], fallback: float = 0.3) -> RewardFn:
    """Stochastic binary reward drawn from Bernoulli(rate).

    Args:
        rates: Dict mapping arm names to success probabilities (0–1).
        fallback: Default rate for unknown arms.

    Returns:
        Function (arm, context) → 0 or 1. Context is ignored.
    """

    def fn(arm: str, context: np.ndarray) -> float:
        p = np.clip(rates.get(arm, fallback), 0.0, 1.0)
        return float(np.random.binomial(1, p))

    return fn


def linear_reward(weights: dict[str, list[float]], fallback: float = 0.4) -> RewardFn:
    """Sigmoid of a linear combination of context features.

    Returns a stochastic binary reward: Bernoulli(sigmoid(w · context)).

    Args:
        weights: Dict mapping arm names to feature weight vectors.
        fallback: Default success rate for unknown arms or mismatched dims.

    Returns:
        Function (arm, context) → 0 or 1.
    """
    w_np = {arm: np.array(w, dtype=np.float64) for arm, w in weights.items()}

    def fn(arm: str, context: np.ndarray) -> float:
        w = w_np.get(arm)
        if w is None or len(w) != len(context):
            return float(np.random.binomial(1, fallback))
        logit = float(np.dot(w, context))
        prob = _sigmoid(logit)
        return float(np.random.binomial(1, prob))

    return fn


def context_free_reward(arm_rates: dict[str, float]) -> RewardFn:
    """Context-free Bernoulli reward based on arm identity alone.

    Convenience alias for categorical_reward — context is ignored.

    Args:
        arm_rates: Dict mapping arm names to success probabilities.

    Returns:
        Function (arm, context) → 0 or 1.
    """
    return categorical_reward(arm_rates, fallback=0.3)


__all__ = [
    "RewardFn",
    "categorical_reward",
    "linear_reward",
    "context_free_reward",
]
