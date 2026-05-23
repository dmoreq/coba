"""Tree-ensemble contextual bandit arm models.

Implements the standard contextual-bandit setting from
"Tree Ensembles for Contextual Bandits" (arXiv:2402.06963) using Random
Forests from scikit-learn. XGBoost/CatBoost-style adapters can share the same
uncertainty utilities later without changing router or bandit APIs.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from coba.policies.base import BaseArmModel
from coba.policies.tree_ensemble_base import (
    _COLD_START_SCORE,
    TreeEnsemblePrediction,
    TreeEnsembleUncertaintyEstimator,
)
from coba.types import Arm


class _RandomForestBanditArmModel(BaseArmModel):
    """Shared storage/retraining logic for Random-Forest bandit arms."""

    def __init__(
        self,
        arm: Arm,
        rng: np.random.Generator,
        n_estimators: int = 50,
        max_depth: int | None = 6,
        min_samples_leaf: int = 1,
        max_obs: int = 1000,
        min_uncertainty: float = 1e-6,
    ) -> None:
        super().__init__(arm, rng)
        if n_estimators < 2:
            raise ValueError("n_estimators must be at least 2 for uncertainty estimation")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        if max_obs < 1:
            raise ValueError("max_obs must be at least 1")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_obs = max_obs
        self._seed = int(rng.integers(0, np.iinfo(np.int32).max))
        self._uncertainty = TreeEnsembleUncertaintyEstimator(min_uncertainty=min_uncertainty)
        self._x: list[np.ndarray] = []
        self._y: list[float] = []
        self._w: list[float] = []
        self.model = self._new_model()

    def _new_model(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self._seed,
            n_jobs=1,
        )

    def update(self, x: np.ndarray, reward: float, weight: float = 1.0) -> None:
        self._x.append(np.asarray(x, dtype=np.float64).copy())
        self._y.append(float(reward))
        self._w.append(float(weight))
        if len(self._y) > self.max_obs:
            overflow = len(self._y) - self.max_obs
            del self._x[:overflow]
            del self._y[:overflow]
            del self._w[:overflow]
        self._refit()

    def update_batch(
        self, x_batch: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
    ) -> None:
        ws = np.ones(len(y), dtype=np.float64) if weights is None else weights
        for xi, yi, wi in zip(x_batch, y, ws):
            self._x.append(np.asarray(xi, dtype=np.float64).copy())
            self._y.append(float(yi))
            self._w.append(float(wi))
        if len(self._y) > self.max_obs:
            overflow = len(self._y) - self.max_obs
            del self._x[:overflow]
            del self._y[:overflow]
            del self._w[:overflow]
        self._refit()

    def _refit(self) -> None:
        if not self._y:
            self.is_fitted = False
            return
        self.model = self._new_model()
        self.model.fit(
            np.asarray(self._x, dtype=np.float64),
            np.asarray(self._y, dtype=np.float64),
            sample_weight=np.asarray(self._w, dtype=np.float64),
        )
        self.is_fitted = True

    def _prediction(self, x: np.ndarray) -> TreeEnsemblePrediction | None:
        if not self.is_fitted:
            return None
        return self._uncertainty.predict(self.model, np.asarray(x, dtype=np.float64))

    def reset(self) -> None:
        self._x.clear()
        self._y.clear()
        self._w.clear()
        self.model = self._new_model()
        self.is_fitted = False

    @property
    def n_obs(self) -> int:
        return len(self._y)


class RandomForestUCBArmModel(_RandomForestBanditArmModel):
    """Random Forest arm model with UCB exploration from tree disagreement."""

    def __init__(
        self,
        arm: Arm,
        rng: np.random.Generator,
        n_estimators: int = 50,
        max_depth: int | None = 6,
        min_samples_leaf: int = 1,
        max_obs: int = 1000,
        min_uncertainty: float = 1e-6,
        alpha: float = 1.0,
    ) -> None:
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        super().__init__(
            arm=arm,
            rng=rng,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_obs=max_obs,
            min_uncertainty=min_uncertainty,
        )
        self.alpha = float(alpha)

    def score(self, x: np.ndarray) -> float:
        prediction = self._prediction(x)
        if prediction is None:
            return _COLD_START_SCORE
        return prediction.ucb(self.alpha)

    def score_decomposed(self, x: np.ndarray) -> tuple[float, float]:
        prediction = self._prediction(x)
        if prediction is None:
            return _COLD_START_SCORE, 0.0
        return prediction.mean, self.alpha * prediction.uncertainty


class RandomForestTSArmModel(_RandomForestBanditArmModel):
    """Random Forest arm model with Thompson-style posterior sampling."""

    def score(self, x: np.ndarray) -> float:
        prediction = self._prediction(x)
        if prediction is None:
            return _COLD_START_SCORE
        return prediction.sample(self.rng)


__all__ = ["RandomForestTSArmModel", "RandomForestUCBArmModel"]
