"""Data generators for continuous action bandit scenarios.

Provides realistic synthetic data for bid pricing optimization and other
continuous action problems.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def generate_bid_pricing_data(
    n_contexts: int = 500,
    n_features: int = 8,
    a_min: float = 0.50,
    a_max: float = 5.00,
    seed: int = 42,
) -> tuple[np.ndarray, Callable[[float, np.ndarray], float]]:
    """Generate realistic bid pricing auction data.

    Models a programmatic advertising auction where:
    - Context: RTB signals (device, geo, user, time, etc.)
    - Action: Bid price in USD
    - Reward: Win probability × (advertiser value - cost)

    The true reward has a bell-shaped curve around an optimal bid price
    that depends on the context. Too low → lose auction. Too high → win
    but overpay.

    Args:
        n_contexts: Number of auction contexts to generate.
        n_features: Context vector dimensionality. Use 8 for standard RTB features.
        a_min: Minimum bid price (default $0.50).
        a_max: Maximum bid price (default $5.00).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (contexts, true_reward_fn) where:
        - contexts: Array shape (n_contexts, n_features)
        - true_reward_fn: Callable(action, context) → scalar reward in [0, 1]
    """
    rng = np.random.default_rng(seed)

    # Generate RTB context features
    if n_features >= 8:
        # Standard RTB features with semantic meaning
        hours = rng.uniform(0, 24, n_contexts)
        time_sin = np.sin(hours / 24 * 2 * np.pi)
        time_cos = np.cos(hours / 24 * 2 * np.pi)
        device_mobile = rng.binomial(1, 0.6, n_contexts).astype(float)
        geo_tier = rng.uniform(0, 1, n_contexts)  # 0=low, 1=premium
        recency_hours = np.clip(rng.exponential(0.3, n_contexts), 0, 1)
        historical_ctr = rng.beta(2, 5, n_contexts)
        auction_floor = rng.uniform(0, 1, n_contexts)  # normalized floor price
        viewability_score = rng.beta(3, 2, n_contexts)

        if n_features == 8:
            contexts = np.column_stack(
                [
                    time_sin,
                    time_cos,
                    device_mobile,
                    geo_tier,
                    recency_hours,
                    historical_ctr,
                    auction_floor,
                    viewability_score,
                ]
            )
        else:
            # Pad with random features if n_features > 8
            contexts = np.column_stack(
                [
                    time_sin,
                    time_cos,
                    device_mobile,
                    geo_tier,
                    recency_hours,
                    historical_ctr,
                    auction_floor,
                    viewability_score,
                ]
            )
            extra_features = rng.standard_normal((n_contexts, n_features - 8))
            contexts = np.column_stack([contexts, extra_features])
    else:
        # Fall back to generic features
        contexts = rng.standard_normal((n_contexts, n_features))

    # Optimal bid price varies by context (learned from features)
    # Base bid around $2.00 with context-dependent adjustments
    optimal_bid = (a_min + a_max) / 2 + 0.5 * np.sin(contexts[:, 0])
    optimal_bid = np.clip(optimal_bid, a_min, a_max)

    def true_reward_fn(action: float, context: np.ndarray) -> float:
        """Compute true reward for (action, context) pair.

        Reward is highest when bid ≈ optimal for that context.
        Models: win_rate × (advertiser_value - cost)
        """
        # Distance from optimal bid (penalize both under/overbidding)
        idx = hash(context.tobytes()) % n_contexts  # Map context back to idx
        optimal = optimal_bid[idx]

        # Bell-shaped reward: peak at optimal bid
        distance = abs(action - optimal)
        max_distance = (a_max - a_min) / 2
        reward = float(np.exp(-(distance**2) / (2 * (max_distance / 3) ** 2)))

        # Add noise
        noise = float(rng.normal(0, 0.05))
        return float(np.clip(reward + noise, 0.0, 1.0))

    return contexts, true_reward_fn


def generate_continuous_linear_data(
    n_contexts: int = 500,
    n_features: int = 8,
    a_min: float = 0.0,
    a_max: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, Callable[[float, np.ndarray], float]]:
    """Generate linear reward model for continuous actions.

    Simple model where reward is a linear function of action + context.
    Useful for testing basic learning without complex interactions.

    Args:
        n_contexts: Number of contexts.
        n_features: Context dimensionality.
        a_min: Action space lower bound.
        a_max: Action space upper bound.
        seed: Random seed.

    Returns:
        Tuple of (contexts, true_reward_fn).
    """
    rng = np.random.default_rng(seed)

    contexts = rng.standard_normal((n_contexts, n_features))

    # Weight vector for linear model
    weights = rng.standard_normal(n_features)
    # Bias term
    bias = 0.5

    def true_reward_fn(action: float, context: np.ndarray) -> float:
        """Linear reward: bias + context @ weights + action effect."""
        linear_context = float(bias + context @ weights)
        # Action benefit decreases toward boundaries (encourage interior)
        action_term = 1.0 - abs(2 * action - (a_min + a_max)) / (a_max - a_min)
        reward = linear_context * 0.3 + action_term * 0.7
        # Normalize to [0, 1]
        reward = (reward + 1) / 2
        noise = float(rng.normal(0, 0.05))
        return float(np.clip(reward + noise, 0.0, 1.0))

    return contexts, true_reward_fn


def generate_biased_action_log(
    n_logs: int = 800,
    n_features: int = 8,
    a_min: float = 0.50,
    a_max: float = 5.00,
    bias_action: float | None = None,
    bias_fraction: float = 0.70,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate historically biased action logs for offline de-biasing demos.

    Simulates a common AdTech scenario: one action (e.g., "aggressive bidding")
    received 70% of traffic due to a legacy policy, but its true reward is lower
    than the under-explored "conservative bidding" strategy.

    Args:
        n_logs: Number of historical log entries.
        n_features: Context dimensionality.
        a_min: Action space lower bound.
        a_max: Action space upper bound.
        bias_action: Which action was overused. None → use midpoint + 0.5.
        bias_fraction: Fraction of traffic to the biased action (0.5-0.95).
        seed: Random seed.

    Returns:
        Tuple of (contexts, actions, propensities, rewards).
    """
    rng = np.random.default_rng(seed)

    contexts = rng.standard_normal((n_logs, n_features))

    # The biased action: perhaps aggressive bidding at the high end
    if bias_action is None:
        bias_action = (a_min + a_max) / 2 + 0.3 * (a_max - a_min)
    else:
        bias_action = float(bias_action)

    # The optimal action: conservative bidding near the low-mid range
    optimal_action = (a_min + a_max) / 2 - 0.2 * (a_max - a_min)
    optimal_action = np.clip(optimal_action, a_min, a_max)

    # Old logging policy: biased heavily toward bias_action
    actions = np.where(
        rng.uniform(0, 1, n_logs) < bias_fraction,
        np.full(n_logs, bias_action),
        rng.uniform(a_min, a_max, n_logs),
    )

    # Propensities: biased action gets high prob, others uniform
    propensities = np.where(
        np.abs(actions - bias_action) < 0.05,
        bias_fraction,
        (1.0 - bias_fraction) / (a_max - a_min),
    )
    propensities = np.clip(propensities, 0.01, 0.99)

    # True rewards: biased action is mediocre, optimal is good
    rewards = np.zeros(n_logs)
    for i in range(n_logs):
        # Distance from optimal
        dist_to_optimal = abs(actions[i] - optimal_action) / (a_max - a_min)
        # Distance from biased
        dist_to_biased = abs(actions[i] - bias_action) / (a_max - a_min)

        # High reward near optimal, low near biased
        reward = 0.8 * np.exp(-(dist_to_optimal**2) / 0.05)
        reward += 0.2 * np.exp(-(dist_to_biased**2) / 0.1) * 0.5  # biased is weak

        rewards[i] = float(np.clip(reward + rng.normal(0, 0.05), 0.0, 1.0))

    return contexts, actions, propensities, rewards
