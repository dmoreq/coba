"""Shared utilities for tree-ensemble contextual bandit policies.

Implements the uncertainty-estimation building block from
"Tree Ensembles for Contextual Bandits" (arXiv:2402.06963): use disagreement
between trees in an ensemble as a predictive uncertainty signal for exploration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


_COLD_START_SCORE: float = float("inf")


@dataclass(frozen=True)
class TreeEnsemblePrediction:
    """Mean and uncertainty extracted from a tree ensemble prediction."""

    mean: float
    uncertainty: float
    tree_predictions: np.ndarray

    def ucb(self, alpha: float) -> float:
        """Return UCB score: mean + alpha * uncertainty."""
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        return float(self.mean + alpha * self.uncertainty)

    def sample(self, rng: np.random.Generator) -> float:
        """Draw a Thompson-style Gaussian sample around the ensemble mean."""
        if self.uncertainty <= 0:
            return float(self.mean)
        return float(rng.normal(self.mean, self.uncertainty))


class TreeEnsembleUncertaintyEstimator:
    """Estimate uncertainty from disagreement across tree predictions.

    The paper's key practical ingredient is that a tree ensemble can expose an
    uncertainty proxy via dispersion across member-tree predictions. This class
    keeps that extraction independent of a specific bandit policy, satisfying
    DRY and single-responsibility principles.
    """

    def __init__(self, min_uncertainty: float = 1e-6) -> None:
        if min_uncertainty < 0:
            raise ValueError(f"min_uncertainty must be non-negative, got {min_uncertainty}")
        self.min_uncertainty = float(min_uncertainty)

    def predict(self, model: Any, x: np.ndarray) -> TreeEnsemblePrediction:
        """Return mean and uncertainty for one feature vector.

        Args:
            model: A fitted scikit-learn style ensemble exposing ``estimators_``.
            x: Context vector, shape (n_features,).
        """
        tree_preds = self.tree_predictions(model, x)
        mean = float(np.mean(tree_preds))
        uncertainty = max(float(np.std(tree_preds, ddof=0)), self.min_uncertainty)
        return TreeEnsemblePrediction(
            mean=mean, uncertainty=uncertainty, tree_predictions=tree_preds
        )

    def tree_predictions(self, model: Any, x: np.ndarray) -> np.ndarray:
        """Return predictions from all member trees for one context vector."""
        estimators = getattr(model, "estimators_", None)
        if estimators is None:
            raise ValueError("model must be fitted and expose estimators_")

        x_2d = np.asarray(x, dtype=np.float64).reshape(1, -1)
        preds: list[float] = []
        for estimator in np.ravel(estimators):
            pred = estimator.predict(x_2d)
            preds.append(float(np.asarray(pred).ravel()[0]))

        if not preds:
            raise ValueError("model exposes no estimators")
        return np.asarray(preds, dtype=np.float64)


__all__ = ["_COLD_START_SCORE", "TreeEnsemblePrediction", "TreeEnsembleUncertaintyEstimator"]
