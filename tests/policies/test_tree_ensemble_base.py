"""Tests for shared tree-ensemble uncertainty utilities."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from coba.policies.tree_ensemble_base import (
    TreeEnsemblePrediction,
    TreeEnsembleUncertaintyEstimator,
)


def _fitted_forest(seed: int = 42) -> RandomForestRegressor:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((40, 3))
    y = x[:, 0] ** 2 + 0.2 * x[:, 1] + rng.normal(0, 0.01, 40)
    model = RandomForestRegressor(n_estimators=7, max_depth=3, random_state=seed)
    model.fit(x, y)
    return model


class TestTreeEnsemblePrediction:
    def test_ucb_adds_scaled_uncertainty(self) -> None:
        pred = TreeEnsemblePrediction(
            mean=1.0,
            uncertainty=0.25,
            tree_predictions=np.array([0.8, 1.2]),
        )
        assert pred.ucb(alpha=2.0) == 1.5

    def test_ucb_rejects_negative_alpha(self) -> None:
        pred = TreeEnsemblePrediction(mean=1.0, uncertainty=0.25, tree_predictions=np.array([1.0]))
        with pytest.raises(ValueError, match="alpha"):
            pred.ucb(alpha=-0.1)

    def test_sample_returns_mean_when_uncertainty_zero(self) -> None:
        pred = TreeEnsemblePrediction(mean=1.2, uncertainty=0.0, tree_predictions=np.array([1.2]))
        assert pred.sample(np.random.default_rng(0)) == 1.2

    def test_sample_is_finite_with_positive_uncertainty(self) -> None:
        pred = TreeEnsemblePrediction(
            mean=1.2, uncertainty=0.3, tree_predictions=np.array([0.9, 1.5])
        )
        sample = pred.sample(np.random.default_rng(0))
        assert np.isfinite(sample)


class TestTreeEnsembleUncertaintyEstimator:
    def test_rejects_negative_min_uncertainty(self) -> None:
        with pytest.raises(ValueError, match="min_uncertainty"):
            TreeEnsembleUncertaintyEstimator(min_uncertainty=-1.0)

    def test_tree_predictions_returns_one_value_per_tree(self) -> None:
        model = _fitted_forest()
        estimator = TreeEnsembleUncertaintyEstimator()
        preds = estimator.tree_predictions(model, np.array([0.1, 0.2, 0.3]))
        assert preds.shape == (7,)
        assert np.all(np.isfinite(preds))

    def test_predict_returns_mean_and_uncertainty(self) -> None:
        model = _fitted_forest()
        estimator = TreeEnsembleUncertaintyEstimator(min_uncertainty=1e-4)
        pred = estimator.predict(model, np.array([0.1, 0.2, 0.3]))
        assert isinstance(pred, TreeEnsemblePrediction)
        assert np.isfinite(pred.mean)
        assert pred.uncertainty >= 1e-4
        assert len(pred.tree_predictions) == 7

    def test_predict_mean_matches_member_average(self) -> None:
        model = _fitted_forest()
        estimator = TreeEnsembleUncertaintyEstimator()
        pred = estimator.predict(model, np.array([0.1, 0.2, 0.3]))
        assert abs(pred.mean - float(np.mean(pred.tree_predictions))) < 1e-12

    def test_unfitted_model_raises_value_error(self) -> None:
        model = RandomForestRegressor(n_estimators=3)
        estimator = TreeEnsembleUncertaintyEstimator()
        with pytest.raises(ValueError, match="estimators_"):
            estimator.predict(model, np.array([0.1, 0.2, 0.3]))
