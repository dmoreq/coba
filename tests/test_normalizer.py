"""Tests for RewardNormalizer."""

import numpy as np
import pytest

from coba.normalizer import RewardNormalizer


class TestRewardNormalizerMinMax:
    def test_first_observation_returns_half(self):
        n = RewardNormalizer(mode="minmax")
        result = n.update_and_normalize(5.0)
        assert result == 0.5

    def test_two_observations_span_zero_one(self):
        n = RewardNormalizer(mode="minmax")
        n.update_and_normalize(0.0)
        result = n.update_and_normalize(100.0)
        assert result == pytest.approx(1.0)

    def test_normalized_values_in_unit_interval(self):
        n = RewardNormalizer(mode="minmax", decay=0.99)
        rewards = [10.0, 20.0, 5.0, 15.0, 25.0, 1.0, 30.0]
        for r in rewards:
            normed = n.update_and_normalize(r)
            assert 0.0 <= normed <= 1.0

    def test_clip_false_allows_outside_range(self):
        n = RewardNormalizer(mode="minmax", clip=False)
        n.update_and_normalize(5.0)
        n.update_and_normalize(10.0)
        # A value below the observed min will produce a negative normed value
        normed = n.normalize(0.0)
        assert normed < 0.0

    def test_normalize_without_update(self):
        n = RewardNormalizer(mode="minmax")
        n.update_and_normalize(0.0)
        n.update_and_normalize(10.0)
        # normalize() uses current stats but doesn't change them
        v1 = n.normalize(5.0)
        v2 = n.normalize(5.0)
        assert v1 == v2

    def test_not_fitted_before_first_observation(self):
        n = RewardNormalizer()
        assert not n.is_fitted
        n.update_and_normalize(1.0)
        assert n.is_fitted

    def test_reset_clears_state(self):
        n = RewardNormalizer()
        n.update_and_normalize(100.0)
        n.reset()
        assert not n.is_fitted

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            RewardNormalizer(mode="invalid")  # type: ignore[arg-type]

    def test_invalid_decay_raises(self):
        with pytest.raises(ValueError, match="decay"):
            RewardNormalizer(decay=1.5)

    def test_non_finite_reward_raises(self):
        n = RewardNormalizer()
        with pytest.raises(ValueError, match="finite"):
            n.update_and_normalize(float("nan"))


class TestRewardNormalizerZScore:
    def test_zscore_after_many_observations_near_zero_mean(self):
        n = RewardNormalizer(mode="zscore", decay=0.95)
        rng = np.random.default_rng(0)
        values = rng.normal(loc=50.0, scale=10.0, size=500)
        normed = [n.update_and_normalize(float(v)) for v in values]
        # After warmup, mean of normalized values should be near 0
        assert abs(np.mean(normed[200:])) < 1.0

    def test_zscore_at_running_mean_is_zero(self):
        """Normalizing the running mean itself must always yield exactly 0."""
        n = RewardNormalizer(mode="zscore", decay=0.99)
        rng = np.random.default_rng(1)
        for v in rng.normal(loc=50.0, scale=5.0, size=300):
            n.update_and_normalize(float(v))
        result = n.normalize(n._mean)
        assert result == pytest.approx(0.0, abs=1e-6)


class TestRewardNormalizerIntegration:
    def test_normalize_then_pass_to_thompson_range(self):
        """Normalized rewards must stay in [0, 1] to be valid for Thompson Sampling."""
        n = RewardNormalizer(mode="minmax")
        rng = np.random.default_rng(42)
        raw_rewards = rng.uniform(50, 500, 200)
        normed = [n.update_and_normalize(float(r)) for r in raw_rewards]
        assert all(0.0 <= v <= 1.0 for v in normed)
