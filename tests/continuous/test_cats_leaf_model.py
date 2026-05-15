"""Tests for CATSLeafModel."""

import numpy as np
import pytest

from coba.continuous.action_tree import ActionLeaf
from coba.policies.cats import CATSLeafModel


class TestCATSLeafModel:
    """Test CATSLeafModel initialization and LinUCB inheritance."""

    def test_model_initialization(self) -> None:
        """CATSLeafModel initializes with leaf and inherits LinUCB."""
        leaf = ActionLeaf(index=5, lo=2.0, hi=3.0, midpoint=2.5)
        model = CATSLeafModel(
            leaf=leaf,
            n_features=8,
            alpha=1.0,
            l2_lambda=1.0,
            gamma=1.0,
        )
        assert model.leaf == leaf
        assert model.arm == 5  # arm identifier is the leaf index
        assert model.n_features == 8
        assert model.alpha == 1.0
        assert not model.is_fitted

    def test_model_update_single_observation(self) -> None:
        """CATSLeafModel.update() works like LinUCBArmModel."""
        leaf = ActionLeaf(index=0, lo=0.0, hi=1.0, midpoint=0.5)
        model = CATSLeafModel(leaf=leaf, n_features=4)

        x = np.array([1.0, 2.0, 3.0, 4.0])
        reward = 0.8
        model.update(x, reward, weight=1.0)

        assert model.is_fitted
        assert model.n_obs == 1

    def test_model_batch_update(self) -> None:
        """CATSLeafModel.update_batch() works like LinUCBArmModel."""
        leaf = ActionLeaf(index=0, lo=0.0, hi=1.0, midpoint=0.5)
        model = CATSLeafModel(leaf=leaf, n_features=4)

        x_batch = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0],
            ]
        )
        y = np.array([0.5, 0.7, 0.9])
        model.update_batch(x_batch, y)

        assert model.is_fitted
        assert model.n_obs == 3

    def test_model_score_returns_float(self) -> None:
        """CATSLeafModel.score() returns a float score."""
        leaf = ActionLeaf(index=0, lo=0.0, hi=1.0, midpoint=0.5)
        model = CATSLeafModel(leaf=leaf, n_features=4)

        # Update with some data first
        x_batch = np.random.default_rng(42).standard_normal((5, 4))
        y = np.random.default_rng(42).uniform(0, 1, 5)
        model.update_batch(x_batch, y)

        # Score should be a positive float
        x = np.array([1.0, 2.0, 3.0, 4.0])
        score = model.score(x)
        assert isinstance(score, float | np.floating)
        assert np.isfinite(score)

    def test_model_score_decomposed(self) -> None:
        """CATSLeafModel.score_decomposed() returns (mean, width) tuple."""
        leaf = ActionLeaf(index=0, lo=0.0, hi=1.0, midpoint=0.5)
        model = CATSLeafModel(leaf=leaf, n_features=4, alpha=1.0)

        # Update with data
        rng = np.random.default_rng(42)
        x_batch = rng.standard_normal((10, 4))
        y = rng.uniform(0, 1, 10)
        model.update_batch(x_batch, y)

        # Decomposed score
        x = np.array([1.0, 2.0, 3.0, 4.0])
        mean, width = model.score_decomposed(x)

        assert isinstance(mean, float | np.floating)
        assert isinstance(width, float | np.floating)
        assert np.isfinite(mean)
        assert np.isfinite(width)
        assert width >= 0.0  # confidence width is always non-negative
        # Full score should equal mean + width
        full_score = model.score(x)
        assert abs(full_score - (mean + width)) < 1e-9

    def test_model_reset(self) -> None:
        """CATSLeafModel.reset() clears learned state."""
        leaf = ActionLeaf(index=0, lo=0.0, hi=1.0, midpoint=0.5)
        model = CATSLeafModel(leaf=leaf, n_features=4)

        # Update
        x = np.array([1.0, 2.0, 3.0, 4.0])
        model.update(x, reward=0.8)
        assert model.is_fitted
        assert model.n_obs == 1

        # Reset
        model.reset()
        assert not model.is_fitted
        assert model.n_obs == 0

    def test_model_clone(self) -> None:
        """CATSLeafModel.clone() creates a deep copy."""
        leaf = ActionLeaf(index=3, lo=1.5, hi=2.0, midpoint=1.75)
        model = CATSLeafModel(leaf=leaf, n_features=4, alpha=0.5)

        # Update original
        x = np.array([1.0, 2.0, 3.0, 4.0])
        model.update(x, reward=0.8)

        # Clone
        cloned = model.clone()

        # Verify clone is independent
        assert cloned.leaf == model.leaf
        assert cloned.n_obs == model.n_obs
        assert cloned.alpha == model.alpha
        assert cloned is not model
        assert cloned._ridge is not model._ridge

    def test_model_different_alphas(self) -> None:
        """CATSLeafModel with different alphas produce different scores."""
        leaf = ActionLeaf(index=0, lo=0.0, hi=1.0, midpoint=0.5)
        model_low_alpha = CATSLeafModel(leaf=leaf, n_features=4, alpha=0.1)
        model_high_alpha = CATSLeafModel(leaf=leaf, n_features=4, alpha=2.0)

        # Train both on same data
        rng = np.random.default_rng(42)
        x_batch = rng.standard_normal((10, 4))
        y = rng.uniform(0, 1, 10)
        model_low_alpha.update_batch(x_batch, y)
        model_high_alpha.update_batch(x_batch, y)

        # Score the same context
        x = np.array([1.0, 2.0, 3.0, 4.0])
        mean_low, width_low = model_low_alpha.score_decomposed(x)
        mean_high, width_high = model_high_alpha.score_decomposed(x)

        # Means should be identical (same ridge solution)
        assert abs(mean_low - mean_high) < 1e-9
        # Widths should differ: higher alpha → wider confidence interval
        assert width_high > width_low

    def test_model_with_ips_weights(self) -> None:
        """CATSLeafModel respects IPS importance weights."""
        leaf = ActionLeaf(index=0, lo=0.0, hi=1.0, midpoint=0.5)
        model = CATSLeafModel(leaf=leaf, n_features=4)

        x = np.array([1.0, 2.0, 3.0, 4.0])
        reward = 0.8
        weight = 2.0  # Observation is weighted 2x
        model.update(x, reward, weight=weight)

        assert model.is_fitted
        # With IPS weighting, the update should affect the model's ridge state

    def test_model_leaf_is_frozen_dataclass(self) -> None:
        """CATSLeafModel stores the frozen ActionLeaf correctly."""
        leaf = ActionLeaf(index=0, lo=0.0, hi=1.0, midpoint=0.5)
        model = CATSLeafModel(leaf=leaf, n_features=4)

        # Leaf is stored and accessible
        assert model.leaf.index == 0
        assert model.leaf.lo == 0.0
        # Leaf itself is frozen (immutable)
        with pytest.raises(AttributeError):
            leaf.index = 99  # type: ignore[misc]
