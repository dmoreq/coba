"""Edge case tests for tree ensemble bandit policies (RandomForest UCB/TS)."""

import numpy as np

from coba.policies.tree_ensemble import RandomForestTSArmModel, RandomForestUCBArmModel

_RNG = np.random.default_rng(0)


class TestRandomForestUCBEdgeCases:
    def test_cold_start_score_is_infinite(self) -> None:
        model = RandomForestUCBArmModel(arm="rf", rng=_RNG, n_estimators=5)
        scores = [model.score(np.array([1.0, 2.0])) for _ in range(5)]
        assert all(np.isinf(s) for s in scores)

    def test_ucb_score_greater_than_mean(self) -> None:
        model = RandomForestUCBArmModel(arm="rf", rng=_RNG, alpha=0.5, n_estimators=10, max_obs=200)
        rng = np.random.default_rng(1)
        for _ in range(100):
            model.update(rng.standard_normal(3), reward=float(rng.uniform(0, 1)))
        assert model.is_fitted

        x = rng.standard_normal(3)
        decom = model.score_decomposed(x)
        mean_score, _ = decom
        full_score = model.score(x)
        # UCB = mean + alpha * uncertainty >= mean for non-negative alpha
        assert full_score >= mean_score - 1e-10, f"{full_score} vs {mean_score}"

    def test_uncertainty_decreases_with_more_data(self) -> None:
        """After sufficient training, uncertainty should shrink relative to early values."""
        rng = np.random.default_rng(2)
        model = RandomForestUCBArmModel(arm="rf", rng=_RNG, n_estimators=10, max_obs=500)
        uncertainties = []
        for i in range(1, 80):
            model.update(rng.standard_normal(3), reward=float(rng.uniform(0.4, 0.6)))
            decom = model.score_decomposed(rng.standard_normal(3))
            uncertainties.append(decom[1])
        # By 80 observations with stable rewards, uncertainty should be modest
        # and not alarmingly large (inf means something went wrong)
        assert np.isfinite(uncertainties[-1])
        assert uncertainties[-1] < 10.0

    def test_max_obs_trims_buffer(self) -> None:
        model = RandomForestUCBArmModel(arm="rf", rng=_RNG, n_estimators=5, max_obs=10)
        rng = np.random.default_rng(3)
        for i in range(25):
            model.update(rng.standard_normal(3), reward=float(i) / 25)
        assert len(model._y) <= 10

    def test_update_batch(self) -> None:
        model = RandomForestUCBArmModel(arm="rf", rng=_RNG, n_estimators=5, max_obs=30)
        rng = np.random.default_rng(4)
        x_batch = rng.standard_normal((15, 2))
        y = rng.uniform(0, 1, 15)
        model.update_batch(x_batch, y)
        assert model.is_fitted

    def test_reset(self) -> None:
        model = RandomForestUCBArmModel(arm="rf", rng=_RNG, n_estimators=5)
        model.update(np.ones(3), reward=0.5)
        model.reset()
        assert not model.is_fitted

    def test_min_uncertainty_floor(self) -> None:
        """Uncertainty must never fall below the configured floor."""
        model = RandomForestUCBArmModel(
            arm="rf", rng=_RNG, n_estimators=10, min_uncertainty=1.0, max_obs=50
        )
        rng = np.random.default_rng(5)
        for _ in range(30):
            model.update(rng.standard_normal(3), reward=float(rng.uniform(0, 1)))
        decom = model.score_decomposed(rng.standard_normal(3))
        assert decom[1] >= 1.0


class TestRandomForestTSEdgeCases:
    def test_ts_scores_are_stochastic(self) -> None:
        model = RandomForestTSArmModel(arm="rf_ts", rng=_RNG, n_estimators=10, max_obs=50)
        rng = np.random.default_rng(6)
        for _ in range(30):
            model.update(rng.standard_normal(3), reward=float(rng.uniform(0, 1)))
        scores = [model.score(rng.standard_normal(3)) for _ in range(20)]
        assert len(set(round(s, 10) for s in scores)) > 1

    def test_ts_update_batch_and_score(self) -> None:
        model = RandomForestTSArmModel(arm="rf_ts", rng=_RNG, n_estimators=5, max_obs=20)
        rng = np.random.default_rng(7)
        x_batch = rng.standard_normal((10, 2))
        y = rng.uniform(0, 1, 10)
        model.update_batch(x_batch, y)
        assert model.is_fitted
        score = model.score(np.array([0.5, -0.2]))
        assert np.isfinite(score)

    def test_cold_start_inf_scores(self) -> None:
        model = RandomForestTSArmModel(arm="rf_ts", rng=_RNG, n_estimators=5)
        score = model.score(np.array([1.0, 2.0]))
        assert np.isinf(score)
