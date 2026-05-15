"""Tests for continuous action data generators."""

import numpy as np

from coba.continuous.simulator import (
    generate_bid_pricing_data,
    generate_biased_action_log,
    generate_continuous_linear_data,
)


class TestBidPricingDataGenerator:
    """Test generate_bid_pricing_data()."""

    def test_bid_pricing_returns_contexts_and_fn(self) -> None:
        """generate_bid_pricing_data() returns contexts and reward function."""
        contexts, reward_fn = generate_bid_pricing_data(n_contexts=100, n_features=8)

        assert contexts.shape == (100, 8)
        assert callable(reward_fn)

    def test_bid_pricing_default_params(self) -> None:
        """generate_bid_pricing_data() has sensible defaults."""
        contexts, reward_fn = generate_bid_pricing_data()
        assert contexts.shape == (500, 8)

    def test_bid_pricing_custom_bounds(self) -> None:
        """generate_bid_pricing_data() respects custom action bounds."""
        contexts, reward_fn = generate_bid_pricing_data(n_contexts=50, a_min=1.0, a_max=10.0)
        assert contexts.shape == (50, 8)
        # Reward function should work with actions in the bounds
        reward = reward_fn(5.0, contexts[0])
        assert 0 <= reward <= 1

    def test_bid_pricing_reward_in_range(self) -> None:
        """Rewards are always in [0, 1]."""
        contexts, reward_fn = generate_bid_pricing_data(n_contexts=50, seed=42)

        for action in np.linspace(0.50, 5.00, 10):
            for i in range(5):
                reward = reward_fn(action, contexts[i])
                assert 0 <= reward <= 1

    def test_bid_pricing_has_optimal_region(self) -> None:
        """Rewards are higher for actions near optimal bid."""
        contexts, reward_fn = generate_bid_pricing_data(n_contexts=100, seed=42)

        # Pick a context and evaluate rewards at different bids
        context = contexts[0]
        low_bid_reward = reward_fn(0.50, context)
        mid_bid_reward = reward_fn(2.50, context)
        high_bid_reward = reward_fn(5.00, context)

        # Mid-range bids should generally be better than extremes
        assert mid_bid_reward >= min(low_bid_reward, high_bid_reward) * 0.8

    def test_bid_pricing_custom_features(self) -> None:
        """generate_bid_pricing_data() supports custom feature dimensions."""
        for n_features in [4, 8, 12]:
            contexts, reward_fn = generate_bid_pricing_data(n_contexts=50, n_features=n_features)
            assert contexts.shape == (50, n_features)


class TestContinuousLinearDataGenerator:
    """Test generate_continuous_linear_data()."""

    def test_linear_returns_contexts_and_fn(self) -> None:
        """generate_continuous_linear_data() returns contexts and reward function."""
        contexts, reward_fn = generate_continuous_linear_data(n_contexts=100)

        assert contexts.shape == (100, 8)
        assert callable(reward_fn)

    def test_linear_reward_in_range(self) -> None:
        """Linear rewards are in [0, 1]."""
        contexts, reward_fn = generate_continuous_linear_data(n_contexts=50, seed=42)

        for action in np.linspace(0.0, 1.0, 10):
            for i in range(5):
                reward = reward_fn(action, contexts[i])
                assert 0 <= reward <= 1, f"Reward {reward} out of range for action {action}"

    def test_linear_custom_bounds(self) -> None:
        """generate_continuous_linear_data() respects custom bounds."""
        contexts, reward_fn = generate_continuous_linear_data(a_min=-1.0, a_max=1.0, n_contexts=50)
        # Test that function works with custom bounds
        reward = reward_fn(0.0, contexts[0])
        assert 0 <= reward <= 1


class TestBiasedActionLogGenerator:
    """Test generate_biased_action_log()."""

    def test_biased_log_returns_four_arrays(self) -> None:
        """generate_biased_action_log() returns (contexts, actions, propensities, rewards)."""
        contexts, actions, propensities, rewards = generate_biased_action_log(n_logs=100)

        assert contexts.shape == (100, 8)
        assert len(actions) == 100
        assert len(propensities) == 100
        assert len(rewards) == 100

    def test_biased_log_default_params(self) -> None:
        """generate_biased_action_log() has sensible defaults."""
        contexts, actions, propensities, rewards = generate_biased_action_log()
        assert contexts.shape == (800, 8)

    def test_biased_log_bias_fraction_respected(self) -> None:
        """generate_biased_action_log() produces mix of biased and random actions."""
        contexts, actions, propensities, rewards = generate_biased_action_log(
            n_logs=1000, bias_fraction=0.70, seed=42
        )

        # Action variance should be lower than uniform (due to bias toward one action)
        action_variance = np.var(actions)
        # Variance should be non-trivial but less than full range
        uniform_variance = ((5.0 - 0.5) ** 2) / 12  # Variance of uniform[0.5, 5.0]
        assert 0 < action_variance < uniform_variance

    def test_biased_log_rewards_in_range(self) -> None:
        """Rewards are always in [0, 1]."""
        contexts, actions, propensities, rewards = generate_biased_action_log(n_logs=100, seed=42)
        assert np.all((rewards >= 0.0) & (rewards <= 1.0))

    def test_biased_log_propensities_in_range(self) -> None:
        """Propensities are in (0, 1)."""
        contexts, actions, propensities, rewards = generate_biased_action_log(n_logs=100, seed=42)
        assert np.all((propensities > 0.0) & (propensities <= 1.0))

    def test_biased_log_custom_bias_action(self) -> None:
        """generate_biased_action_log() accepts custom biased action."""
        contexts, actions, propensities, rewards = generate_biased_action_log(
            n_logs=100, bias_action=1.0, a_min=0.5, a_max=5.0, seed=42
        )
        # Some actions should be near 1.0 (the bias action)
        near_bias = np.sum(np.abs(actions - 1.0) < 0.1)
        assert near_bias > 0


class TestGeneratorConsistency:
    """Test consistency and seeding across generators."""

    def test_seed_reproducibility_bid_pricing(self) -> None:
        """Same seed produces identical bid pricing data."""
        ctx1, fn1 = generate_bid_pricing_data(n_contexts=50, seed=42)
        ctx2, fn2 = generate_bid_pricing_data(n_contexts=50, seed=42)

        assert np.allclose(ctx1, ctx2)
        # Test reward functions with same contexts
        for i in range(5):
            r1 = fn1(2.5, ctx1[i])
            r2 = fn2(2.5, ctx2[i])
            assert abs(r1 - r2) < 0.01  # Approx equal (noise may differ)

    def test_seed_reproducibility_biased_log(self) -> None:
        """Same seed produces identical biased logs."""
        ctx1, act1, prop1, rew1 = generate_biased_action_log(n_logs=100, seed=42)
        ctx2, act2, prop2, rew2 = generate_biased_action_log(n_logs=100, seed=42)

        assert np.allclose(ctx1, ctx2)
        assert np.allclose(act1, act2)
        assert np.allclose(prop1, prop2)
        assert np.allclose(rew1, rew2)

    def test_different_seeds_produce_different_data(self) -> None:
        """Different seeds produce different data."""
        ctx1, _ = generate_bid_pricing_data(n_contexts=100, seed=42)
        ctx2, _ = generate_bid_pricing_data(n_contexts=100, seed=99)

        # Contexts should be different (with high probability)
        assert not np.allclose(ctx1, ctx2)
