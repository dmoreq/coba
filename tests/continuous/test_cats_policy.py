"""Tests for CATSPolicy core algorithm."""

import numpy as np
import pytest

from coba.continuous.action_tree import BinaryActionTree
from coba.continuous.policy import CATSPolicy
from coba.continuous.schemas import ContinuousDecision


class TestCATSPolicyInitialization:
    """Test CATSPolicy construction and initialization."""

    def test_policy_creation_with_defaults(self) -> None:
        """CATSPolicy initializes with sensible defaults."""
        tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=4)
        policy = CATSPolicy(tree=tree, n_features=8)
        assert policy.n_leaves == 16
        assert not policy.is_fitted
        assert policy._total_pulls == 0

    def test_policy_creation_with_custom_params(self) -> None:
        """CATSPolicy accepts custom hyperparameters."""
        tree = BinaryActionTree(a_min=0.5, a_max=5.0, depth=5)
        policy = CATSPolicy(
            tree=tree,
            n_features=6,
            alpha=2.0,
            l2_lambda=0.5,
            gamma=0.9,
            seed=123,
        )
        assert policy._alpha == 2.0
        assert policy._l2_lambda == 0.5
        assert policy._gamma == 0.9

    def test_policy_leaf_models_initialized(self) -> None:
        """CATSPolicy creates one CATSLeafModel per leaf."""
        tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=3)
        policy = CATSPolicy(tree=tree, n_features=4)
        assert len(policy._leaf_models) == 8
        for leaf_idx in range(8):
            assert leaf_idx in policy._leaf_models


class TestCATSPolicyDecision:
    """Test CATS action selection (decide)."""

    def test_decide_returns_continuous_decision(self) -> None:
        """decide() returns a valid ContinuousDecision."""
        tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=3)
        policy = CATSPolicy(tree=tree, n_features=4, seed=42)

        x = np.array([0.1, 0.2, 0.3, 0.4])
        decision = policy.decide(x)

        assert isinstance(decision, ContinuousDecision)
        assert decision.leaf_lo <= decision.chosen_action <= decision.leaf_hi
        assert decision.propensity > 0
        assert decision.leaf_index in range(8)

    def test_decide_action_within_bounds(self) -> None:
        """decide() returns action within tree bounds."""
        tree = BinaryActionTree(a_min=0.5, a_max=5.0, depth=4)
        policy = CATSPolicy(tree=tree, n_features=5)

        rng = np.random.default_rng(42)
        for _ in range(20):
            x = rng.standard_normal(5)
            decision = policy.decide(x)
            assert tree.a_min <= decision.chosen_action <= tree.a_max

    def test_decide_increments_total_pulls(self) -> None:
        """decide() increments _total_pulls counter."""
        tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=2)
        policy = CATSPolicy(tree=tree, n_features=3)

        assert policy._total_pulls == 0
        x = np.array([0.1, 0.2, 0.3])
        for i in range(5):
            policy.decide(x)
            assert policy._total_pulls == i + 1

    def test_decide_propensity_positive(self) -> None:
        """decide() propensity is always positive."""
        tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=3)
        policy = CATSPolicy(tree=tree, n_features=4, seed=99)

        rng = np.random.default_rng(42)
        for _ in range(10):
            x = rng.standard_normal(4)
            decision = policy.decide(x)
            assert decision.propensity > 0

    def test_decide_context_shape_validation(self) -> None:
        """decide() validates context shape."""
        tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=2)
        policy = CATSPolicy(tree=tree, n_features=5)

        # Wrong shape
        with pytest.raises(ValueError, match="context shape mismatch"):
            policy.decide(np.array([0.1, 0.2, 0.3]))  # 3-dim context, need 5

    def test_decide_context_finite_validation(self) -> None:
        """decide() rejects non-finite contexts."""
        tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=2)
        policy = CATSPolicy(tree=tree, n_features=3)

        # NaN in context
        with pytest.raises(ValueError, match="non-finite"):
            policy.decide(np.array([0.1, np.nan, 0.3]))


class TestCATSPolicyScoring:
    """Test leaf scoring."""

    def test_score_all_leaves_returns_dict(self) -> None:
        """score_all_leaves() returns dict of all leaf scores."""
        tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=3)
        policy = CATSPolicy(tree=tree, n_features=4)

        x = np.array([0.1, 0.2, 0.3, 0.4])
        scores = policy.score_all_leaves(x)

        assert isinstance(scores, dict)
        assert len(scores) == 8
        assert all(isinstance(idx, int) for idx in scores.keys())
        assert all(isinstance(score, float | np.floating) for score in scores.values())

    def test_score_all_leaves_cold_start(self) -> None:
        """score_all_leaves() works on unfitted policy (cold start)."""
        tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=2)
        policy = CATSPolicy(tree=tree, n_features=3)

        x = np.array([0.1, 0.2, 0.3])
        scores = policy.score_all_leaves(x)

        # Cold start: all models untrained, should have default scores
        assert len(scores) == 4
        assert all(np.isfinite(s) for s in scores.values())


class TestCATSPolicyUpdate:
    """Test single observation update."""

    def test_update_single_observation(self) -> None:
        """update() processes single (context, action, reward) tuple."""
        tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=3)
        policy = CATSPolicy(tree=tree, n_features=4)

        x = np.array([0.1, 0.2, 0.3, 0.4])
        action = 0.5
        reward = 0.8

        assert not policy.is_fitted
        policy.update(x, action, reward)
        assert policy.is_fitted

    def test_update_assigns_to_correct_leaf(self) -> None:
        """update() assigns observation to the leaf containing the action."""
        tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=2)  # 4 leaves
        policy = CATSPolicy(tree=tree, n_features=3, seed=42)

        x = np.array([0.1, 0.2, 0.3])
        action = 0.6  # Should be in leaf 2 ([0.5, 0.75))
        reward = 0.7

        policy.update(x, action, reward)

        # Check that the correct leaf was updated
        leaf = tree.leaf_for_action(action)
        assert policy._leaf_models[leaf.index].is_fitted

    def test_update_respects_ips_weight(self) -> None:
        """update() applies IPS weight correctly."""
        tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=2)
        policy = CATSPolicy(tree=tree, n_features=3)

        x = np.array([0.1, 0.2, 0.3])
        action = 0.3
        reward = 0.8
        propensity = 0.5

        # Update with non-uniform propensity
        policy.update(x, action, reward, propensity=propensity)
        assert policy.is_fitted

    def test_update_reward_validation(self) -> None:
        """update() validates reward is finite."""
        tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=2)
        policy = CATSPolicy(tree=tree, n_features=3)

        x = np.array([0.1, 0.2, 0.3])
        action = 0.5

        with pytest.raises(ValueError, match="reward must be finite"):
            policy.update(x, action, np.nan)

    def test_update_propensity_validation(self) -> None:
        """update() validates propensity is in (0, 1.0]."""
        tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=2)
        policy = CATSPolicy(tree=tree, n_features=3)

        x = np.array([0.1, 0.2, 0.3])
        action = 0.5
        reward = 0.8

        # Zero propensity
        with pytest.raises(ValueError, match="propensity must be in"):
            policy.update(x, action, reward, propensity=0.0)

        # Out of range propensity
        with pytest.raises(ValueError, match="propensity must be in"):
            policy.update(x, action, reward, propensity=1.5)


class TestCATSPolicyBatchUpdate:
    """Test batch fit_batch() method."""

    def test_fit_batch_basic(self) -> None:
        """fit_batch() processes batch of observations."""
        tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=3)
        policy = CATSPolicy(tree=tree, n_features=4)

        rng = np.random.default_rng(42)
        n_samples = 20
        contexts = rng.standard_normal((n_samples, 4))
        actions = rng.uniform(0.0, 1.0, n_samples)
        rewards = rng.uniform(0.0, 1.0, n_samples)

        assert not policy.is_fitted
        policy.fit_batch(contexts, actions, rewards)
        assert policy.is_fitted
        assert policy._total_pulls == n_samples

    def test_fit_batch_with_propensities(self) -> None:
        """fit_batch() respects propensity weights."""
        tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=2)
        policy = CATSPolicy(tree=tree, n_features=3)

        rng = np.random.default_rng(42)
        n_samples = 10
        contexts = rng.standard_normal((n_samples, 3))
        actions = rng.uniform(0.0, 1.0, n_samples)
        rewards = rng.uniform(0.0, 1.0, n_samples)
        propensities = rng.uniform(0.1, 1.0, n_samples)

        policy.fit_batch(contexts, actions, rewards, propensities)
        assert policy.is_fitted

    def test_fit_batch_shape_validation(self) -> None:
        """fit_batch() validates input shapes."""
        tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=2)
        policy = CATSPolicy(tree=tree, n_features=4)

        contexts = np.random.standard_normal((10, 3))  # Wrong feature dim
        actions = np.random.uniform(0.0, 1.0, 10)
        rewards = np.random.uniform(0.0, 1.0, 10)

        with pytest.raises(ValueError, match="contexts shape mismatch"):
            policy.fit_batch(contexts, actions, rewards)

    def test_fit_batch_length_validation(self) -> None:
        """fit_batch() validates input lengths match."""
        tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=2)
        policy = CATSPolicy(tree=tree, n_features=3)

        contexts = np.random.standard_normal((10, 3))
        actions = np.random.uniform(0.0, 1.0, 5)  # Wrong length
        rewards = np.random.uniform(0.0, 1.0, 10)

        with pytest.raises(ValueError, match="same length"):
            policy.fit_batch(contexts, actions, rewards)


class TestCATSPolicyUtilities:
    """Test helper methods and properties."""

    def test_get_leaf_stats(self) -> None:
        """get_leaf_stats() returns per-leaf statistics."""
        tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=2)
        policy = CATSPolicy(tree=tree, n_features=3)

        # Update a few observations
        rng = np.random.default_rng(42)
        contexts = rng.standard_normal((5, 3))
        actions = rng.uniform(0.0, 1.0, 5)
        rewards = rng.uniform(0.0, 1.0, 5)
        policy.fit_batch(contexts, actions, rewards)

        stats = policy.get_leaf_stats()
        assert len(stats) == 4  # 2^2 leaves
        for leaf_idx, leaf_stats in stats.items():
            assert "n_obs" in leaf_stats
            assert "leaf_lo" in leaf_stats
            assert "leaf_hi" in leaf_stats
            assert "midpoint" in leaf_stats

    def test_reset_clears_state(self) -> None:
        """reset() clears all learned state."""
        tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=2)
        policy = CATSPolicy(tree=tree, n_features=3)

        # Train
        x = np.array([0.1, 0.2, 0.3])
        policy.update(x, 0.5, 0.8)
        assert policy.is_fitted
        assert policy._total_pulls == 1

        # Reset
        policy.reset()
        assert not policy.is_fitted
        assert policy._total_pulls == 0

    def test_repr(self) -> None:
        """__repr__ produces informative string."""
        tree = BinaryActionTree(a_min=0.5, a_max=5.0, depth=4)
        policy = CATSPolicy(tree=tree, n_features=8, alpha=1.5)

        repr_str = repr(policy)
        assert "CATSPolicy" in repr_str
        assert "n_features=8" in repr_str
        assert "alpha=1.5" in repr_str
