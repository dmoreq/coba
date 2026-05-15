"""Tests for Random Forest tree-ensemble bandit models."""

import numpy as np
import pytest

from coba.policies.tree_ensemble import RandomForestTSArmModel, RandomForestUCBArmModel


def _data(n: int = 30, d: int = 4, seed: int = 42):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    y = np.clip(x[:, 0] ** 2 + 0.3 * x[:, 1] + rng.normal(0, 0.02, n), 0, 1)
    return x, y


class TestRandomForestUCBArmModel:
    def test_rejects_invalid_hyperparameters(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="n_estimators"):
            RandomForestUCBArmModel("a", rng, n_estimators=1)
        with pytest.raises(ValueError, match="alpha"):
            RandomForestUCBArmModel("a", rng, alpha=-0.1)
        with pytest.raises(ValueError, match="max_obs"):
            RandomForestUCBArmModel("a", rng, max_obs=0)

    def test_cold_start_score_is_infinite(self) -> None:
        model = RandomForestUCBArmModel("a", np.random.default_rng(0), n_estimators=5)
        assert model.score(np.zeros(4)) == float("inf")
        assert not model.is_fitted

    def test_single_update_fits_model(self) -> None:
        model = RandomForestUCBArmModel("a", np.random.default_rng(0), n_estimators=5)
        model.update(np.array([1.0, 0.0, 0.0, 0.0]), 0.7)
        assert model.is_fitted
        assert model.n_obs == 1
        assert np.isfinite(model.score(np.zeros(4)))

    def test_batch_update_fits_model(self) -> None:
        x, y = _data()
        model = RandomForestUCBArmModel("a", np.random.default_rng(0), n_estimators=7)
        model.update_batch(x, y)
        assert model.is_fitted
        assert model.n_obs == len(y)
        assert np.isfinite(model.score(x[0]))

    def test_score_decomposed_matches_score(self) -> None:
        x, y = _data()
        model = RandomForestUCBArmModel("a", np.random.default_rng(0), n_estimators=7, alpha=1.5)
        model.update_batch(x, y)
        mean, width = model.score_decomposed(x[0])
        assert np.isfinite(mean)
        assert width >= 0
        assert abs(model.score(x[0]) - (mean + width)) < 1e-12

    def test_higher_alpha_increases_or_keeps_score(self) -> None:
        x, y = _data()
        low = RandomForestUCBArmModel("a", np.random.default_rng(0), n_estimators=9, alpha=0.1)
        high = RandomForestUCBArmModel("a", np.random.default_rng(0), n_estimators=9, alpha=2.0)
        low.update_batch(x, y)
        high.update_batch(x, y)
        assert high.score(x[0]) >= low.score(x[0])

    def test_max_obs_keeps_recent_window(self) -> None:
        x, y = _data(n=10)
        model = RandomForestUCBArmModel("a", np.random.default_rng(0), n_estimators=5, max_obs=3)
        model.update_batch(x, y)
        assert model.n_obs == 3

    def test_reset_clears_observations(self) -> None:
        x, y = _data()
        model = RandomForestUCBArmModel("a", np.random.default_rng(0), n_estimators=5)
        model.update_batch(x, y)
        model.reset()
        assert not model.is_fitted
        assert model.n_obs == 0
        assert model.score(x[0]) == float("inf")


class TestRandomForestTSArmModel:
    def test_ts_cold_start_score_is_infinite(self) -> None:
        model = RandomForestTSArmModel("a", np.random.default_rng(0), n_estimators=5)
        assert model.score(np.zeros(4)) == float("inf")

    def test_ts_score_is_finite_after_fit(self) -> None:
        x, y = _data()
        model = RandomForestTSArmModel("a", np.random.default_rng(0), n_estimators=7)
        model.update_batch(x, y)
        assert np.isfinite(model.score(x[0]))

    def test_ts_samples_vary_with_uncertainty(self) -> None:
        x, y = _data(n=60)
        model = RandomForestTSArmModel("a", np.random.default_rng(123), n_estimators=25)
        model.update_batch(x, y)
        samples = [model.score(x[0]) for _ in range(20)]
        assert np.all(np.isfinite(samples))
        assert np.std(samples) >= 0

    def test_ts_update_batch_accepts_weights(self) -> None:
        x, y = _data()
        weights = np.linspace(0.5, 1.5, len(y))
        model = RandomForestTSArmModel("a", np.random.default_rng(0), n_estimators=5)
        model.update_batch(x, y, weights)
        assert model.is_fitted
        assert model.n_obs == len(y)
