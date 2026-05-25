"""Property tests for RewardNormalizer (z-score and minmax modes).

Verifies mathematical invariants: z-score of the mean is zero, minmax
maps to [0,1] correctly, clipping behavior, and edge cases like constant
streams and extreme values.
"""

from __future__ import annotations

import numpy as np
import pytest

from coba.normalizer import RewardNormalizer


class TestZScoreProperties:
    """Z-score normalization mathematical properties."""

    def test_zscore_of_running_mean_is_exactly_zero(self) -> None:
        """After feeding a stream, normalize(running_mean) must always be 0.0."""
        n = RewardNormalizer(mode="zscore", decay=0.99)
        rng = np.random.default_rng(0)
        for v in rng.normal(loc=50.0, scale=10.0, size=300):
            n.update_and_normalize(float(v))
        result = n.normalize(n._mean)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_zscore_converges_to_near_zero_mean(self) -> None:
        """After enough warmup, the mean of z-scored values must be near zero."""
        n = RewardNormalizer(mode="zscore", decay=0.99)
        rng = np.random.default_rng(1)
        values = rng.normal(loc=100.0, scale=25.0, size=500)
        normed = [n.update_and_normalize(float(v)) for v in values]
        mean_after_warmup = np.mean(normed[200:])
        assert abs(mean_after_warmup) < 0.5

    def test_zscore_with_constant_stream_converges(self) -> None:
        """All identical values → mean converges to that value.
        Normalizing the running mean itself must always yield exactly 0."""
        n = RewardNormalizer(mode="zscore", decay=0.95)
        for _ in range(500):
            n.update_and_normalize(42.0)
        # The running mean has converged near 42.0; normalizing the mean is zero
        result = n.normalize(n._mean)
        assert result == pytest.approx(0.0, abs=1e-6)
        # A different value far from the converged mean
        result2 = n.normalize(100.0)
        # With near-zero std, this will be very large (but finite)
        assert np.isfinite(result2)

    def test_em_decay_gives_higher_weight_to_recent(self) -> None:
        """After a distribution shift, the running mean must move toward the new
        values. A lower decay (e.g., 0.9) must adapt faster than higher (0.999)."""
        n_fast = RewardNormalizer(mode="zscore", decay=0.9)
        n_slow = RewardNormalizer(mode="zscore", decay=0.999)

        # Feed both 500 values around 0.0
        for _ in range(500):
            n_fast.update_and_normalize(0.0)
            n_slow.update_and_normalize(0.0)

        # Then shift to 10.0 for 50 steps
        for _ in range(50):
            n_fast.update_and_normalize(10.0)
            n_slow.update_and_normalize(10.0)

        # Fast adapter's mean should be closer to 10.0
        assert n_fast._mean > n_slow._mean

    def test_negative_rewards(self) -> None:
        """Z-score normalization works with negative reward values."""
        n = RewardNormalizer(mode="zscore", decay=0.99)
        values = [-10.0, -5.0, 0.0, 5.0, 10.0]
        for v in values:
            result = n.update_and_normalize(float(v))
            assert np.isfinite(result)

    def test_is_fitted_after_first_observation(self) -> None:
        n = RewardNormalizer(mode="zscore")
        assert not n.is_fitted
        n.update_and_normalize(1.0)
        assert n.is_fitted

    def test_reset_clears_zscore_state(self) -> None:
        n = RewardNormalizer(mode="zscore")
        for _ in range(50):
            n.update_and_normalize(1.0)
        n.reset()
        assert not n.is_fitted
        assert n._n == 0
        assert n._mean == 0.0


class TestMinMaxProperties:
    """Min-max normalization mathematical properties."""

    def test_first_observation_returns_half(self) -> None:
        """The very first observation always normalizes to 0.5 (min==max → span=0)."""
        n = RewardNormalizer(mode="minmax")
        assert n.update_and_normalize(5.0) == 0.5
        # Second observation with different value opens the span
        assert n.update_and_normalize(10.0) != 0.5  # now normalized by span

    def test_two_distinct_values_span_full_range(self) -> None:
        """After feeding [0, 100], normalize(0)=0, normalize(100)=1."""
        n = RewardNormalizer(mode="minmax", clip=False)
        n.update_and_normalize(0.0)
        n.update_and_normalize(100.0)
        assert n.normalize(0.0) == pytest.approx(0.0)
        assert n.normalize(100.0) == pytest.approx(1.0)
        assert n.normalize(50.0) == pytest.approx(0.5)

    def test_clip_true_caps_at_bounds(self) -> None:
        """Values beyond observed min/max must be clamped to [0, 1] when clip=True."""
        n = RewardNormalizer(mode="minmax", clip=True)
        n.update_and_normalize(10.0)  # min=max=10 → returns 0.5
        n.update_and_normalize(20.0)  # now min=10, max=20
        assert n.normalize(5.0) == 0.0  # below min → clipped to 0
        assert n.normalize(30.0) == 1.0  # above max → clipped to 1

    def test_clip_false_allows_extrapolation(self) -> None:
        """When clip=False, values outside observed range must produce
        normalized values outside [0, 1]."""
        n = RewardNormalizer(mode="minmax", clip=False)
        n.update_and_normalize(10.0)
        n.update_and_normalize(20.0)
        assert n.normalize(5.0) < 0.0
        assert n.normalize(30.0) > 1.0

    def test_normalize_without_update_does_not_change_stats(self) -> None:
        """normalize() must return the correct value without modifying running stats."""
        n = RewardNormalizer(mode="minmax")
        n.update_and_normalize(10.0)
        n.update_and_normalize(20.0)
        n_after_first = n._n
        result1 = n.normalize(15.0)
        result2 = n.normalize(15.0)
        assert result1 == result2
        assert n._n == n_after_first  # stats unchanged

    def test_span_near_zero_returns_half(self) -> None:
        """When max ≈ min (span < 1e-12), normalize must return 0.5 for any input."""
        n = RewardNormalizer(mode="minmax")
        n.update_and_normalize(100.0)
        # Only one value seen → span is 0 → normalize returns 0.5
        assert n.normalize(0.0) == 0.5
        assert n.normalize(100.0) == 0.5

    def test_negative_rewards_minmax(self) -> None:
        """Minmax normalization with negative and positive rewards works correctly."""
        n = RewardNormalizer(mode="minmax", clip=False)
        n.update_and_normalize(-10.0)
        n.update_and_normalize(10.0)
        assert n.normalize(-10.0) == pytest.approx(0.0)
        assert n.normalize(10.0) == pytest.approx(1.0)
        assert n.normalize(0.0) == pytest.approx(0.5)

    def test_minmax_adapts_to_expanding_range(self) -> None:
        """As the reward range expands, min and max must track the new extremes."""
        n = RewardNormalizer(mode="minmax", decay=0.9, clip=False)
        n.update_and_normalize(0.0)
        n.update_and_normalize(10.0)
        # Expand range upward
        for _ in range(10):
            n.update_and_normalize(50.0)
        assert n._max > 30.0  # max has moved toward 50
        # Expand range downward
        for _ in range(10):
            n.update_and_normalize(-20.0)
        assert n._min < -5.0  # min has moved toward -20


class TestNormalizerEdgeCases:
    """Edge cases for both modes."""

    def test_very_large_reward(self) -> None:
        """Extremely large rewards must not cause overflow."""
        n = RewardNormalizer(mode="minmax", clip=False)
        vals = [1e12, 1e13, 1e11, 5e12]
        for v in vals:
            result = n.update_and_normalize(v)
            assert np.isfinite(result)

    def test_very_small_reward(self) -> None:
        """Extremely small (close to zero) rewards must not cause underflow."""
        n = RewardNormalizer(mode="minmax", clip=False)
        vals = [1e-12, 1e-13, 1e-11]
        for v in vals:
            result = n.update_and_normalize(v)
            assert np.isfinite(result)

    def test_mixed_large_and_small(self) -> None:
        """Massive scale differences must still produce finite output."""
        n = RewardNormalizer(mode="zscore", decay=0.95)
        vals = [1e-10, 1e10, 0.0, 1.0]
        for v in vals:
            result = n.update_and_normalize(v)
            assert np.isfinite(result)
