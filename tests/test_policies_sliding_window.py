"""Tests for SlidingWindowLinUCB arm model."""

import numpy as np
import pytest

from coba.policies.linucb_sw import SlidingWindowLinUCBArmModel


class TestSlidingWindowLinUCB:
    def test_window_grows_until_full(self) -> None:
        """Fewer than window_size observations → all retained in buffer."""
        window = 10
        model = SlidingWindowLinUCBArmModel(arm="sw", n_features=3, window_size=window)
        rng = np.random.default_rng(0)
        for i in range(5):
            model.update(rng.standard_normal(3), reward=float(i) / 10)
        assert len(model._buffer) == 5
        assert model.n_obs == 5

    def test_window_evicts_oldest(self) -> None:
        """More than window_size observations → buffer size capped, oldest dropped."""
        window = 5
        model = SlidingWindowLinUCBArmModel(arm="sw", n_features=2, window_size=window)
        for i in range(10):
            model.update(np.array([float(i), 0.0]), reward=float(i) / 10)
        assert len(model._buffer) == window
        # Buffer should contain the 5 most recent observations (indices 5-9)
        first_x_in_buffer = model._buffer[0][0]
        assert first_x_in_buffer[0] == pytest.approx(5.0)

    def test_refit_produces_different_beta_than_unwindowed(self) -> None:
        """With an abrupt change in the reward signal and a small window,
        the sliding-window model adapts faster than a full-history model."""
        rng = np.random.default_rng(0)
        n_features = 2
        window = 10

        sw_model = SlidingWindowLinUCBArmModel(arm="sw", n_features=n_features, window_size=window)
        full_model = SlidingWindowLinUCBArmModel(
            arm="full", n_features=n_features, window_size=10000
        )

        # Phase 1: reward driven by x[0] > 0
        for _ in range(20):
            x = rng.standard_normal(n_features)
            reward = 1.0 if x[0] > 0 else 0.0
            sw_model.update(x, reward)
            full_model.update(x, reward)

        # Phase 2: reward reverses — driven by x[0] < 0
        for _ in range(20):
            x = rng.standard_normal(n_features)
            reward = 1.0 if x[0] < 0 else 0.0
            sw_model.update(x, reward)
            full_model.update(x, reward)

        # After phase 2, SW model should have adapted (its buffer only has recent data)
        x_pos = np.array([1.0, 0.0])
        x_neg = np.array([-1.0, 0.0])

        sw_score_pos = sw_model.score(x_pos)
        sw_score_neg = sw_model.score(x_neg)
        full_score_pos = full_model.score(x_pos)
        full_score_neg = full_model.score(x_neg)

        # SW model: x_neg should score higher (reversed pattern)
        # Full model: still biased by phase 1, x_pos likely scores higher
        # At minimum, the gap should differ
        sw_gap = sw_score_neg - sw_score_pos
        full_gap = full_score_neg - full_score_pos
        assert sw_gap > full_gap, (
            f"SW gap={sw_gap:.4f}, Full gap={full_gap:.4f} — SW didn't adapt faster"
        )

    def test_window_size_of_one(self) -> None:
        """window_size=1: only the latest observation matters. Edge case."""
        model = SlidingWindowLinUCBArmModel(arm="sw", n_features=2, window_size=1)
        x1 = np.array([1.0, 0.0])
        x2 = np.array([0.0, 1.0])

        model.update(x1, reward=1.0)
        model.update(x2, reward=0.0)

        # After window=1, only x2 is in buffer → score(x2) should reflect 0 reward
        score1 = model.score(x1)
        score2 = model.score(x2)
        assert np.isfinite(score1)
        assert np.isfinite(score2)

    def test_batch_update_respects_window(self) -> None:
        """Batch update with more samples than window_size truncates correctly."""
        window = 3
        model = SlidingWindowLinUCBArmModel(arm="sw", n_features=2, window_size=window)
        rng = np.random.default_rng(0)
        x_batch = rng.standard_normal((10, 2))
        y = rng.uniform(0, 1, 10)
        model.update_batch(x_batch, y)
        assert len(model._buffer) == window

    def test_reset_clears_buffer(self) -> None:
        model = SlidingWindowLinUCBArmModel(arm="sw", n_features=2, window_size=5)
        model.update(np.array([1.0, 0.0]), reward=0.5)
        model.reset()
        assert not model.is_fitted
        assert len(model._buffer) == 0
        assert model.n_obs == 0
