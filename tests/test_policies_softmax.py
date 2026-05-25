"""Tests for SoftmaxArmModel policy."""

import numpy as np

from coba.policies.softmax import SoftmaxArmModel


class TestSoftmaxArmModel:
    def test_cold_start_scores(self) -> None:
        """Before any update, Softmax returns +inf to guarantee exploration."""
        model = SoftmaxArmModel(arm="test", n_features=3, tau=1.0)
        x = np.array([1.0, 0.5, -0.3])
        score = model.score(x)
        assert np.isinf(score)

    def test_score_order_matches_reward_order(self) -> None:
        """After training with known rewards, the trained model predicts higher reward
        for inputs correlated with high reward."""
        model = SoftmaxArmModel(arm="test", n_features=2, tau=1.0)
        # Train with a simple linear pattern: reward ≈ x[0]
        x_mat = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]])
        y_vec = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        model.update_batch(x_mat, y_vec)
        assert model.is_fitted
        # Higher x[0] → higher expected reward → higher softmax score
        assert model.score(np.array([5.0, 0.0])) > model.score(np.array([1.0, 0.0]))

    def test_tau_effect_on_exploration(self) -> None:
        """tau=0.1 (cold) gives steeper score separation than tau=10.0 (flat)."""
        rng = np.random.default_rng(0)
        model_cold = SoftmaxArmModel(arm="cold", n_features=2, tau=0.1)
        model_flat = SoftmaxArmModel(arm="flat", n_features=2, tau=10.0)

        # Train both with a clear linear pattern
        x_mat = rng.standard_normal((30, 2))
        y = x_mat[:, 0] * 0.5 + 0.5  # reward increases with x[0]
        model_cold.update_batch(x_mat, y.clip(0, 1))
        model_flat.update_batch(x_mat, y.clip(0, 1))

        # For a point with high x[0] (high expected reward), tau=0.1 amplifies
        x_high = np.array([3.0, 0.0])
        assert model_cold.score(x_high) > model_flat.score(x_high)

    def test_update_batch(self) -> None:
        model = SoftmaxArmModel(arm="batch", n_features=2, tau=1.0)
        rng = np.random.default_rng(0)
        x_mat = rng.standard_normal((10, 2))
        y = rng.uniform(0, 1, 10)
        model.update_batch(x_mat, y)
        assert model.is_fitted

    def test_reset(self) -> None:
        model = SoftmaxArmModel(arm="reset", n_features=2)
        model.update(np.array([1.0, 0.0]), reward=0.8)
        model.reset()
        assert not model.is_fitted
        assert model.n_obs == 0

    def test_weighted_update(self) -> None:
        model_unweighted = SoftmaxArmModel(arm="uw", n_features=2, tau=1.0)
        model_weighted = SoftmaxArmModel(arm="w", n_features=2, tau=1.0)
        x = np.array([1.0, 0.5])
        model_unweighted.update(x, 0.5, weight=1.0)
        model_weighted.update(x, 0.5, weight=3.0)
        assert model_unweighted.score(x) != model_weighted.score(x)
