# Meta-Heuristics (Scikit-learn Wrappers)

## 1. Overview

* **Type**: Non-linear / Meta-algorithm
* **Classes**: `EpsilonGreedyArmModel`, `BootstrappedTSArmModel`, `BootstrappedUCBArmModel` (`policies/sklearn_models.py`)
* **Best for**: Complex non-linear feature interactions (e.g., categorical features, time-of-day interactions) where a simple linear model underfits.

## 2. How It Works

These policies act as wrappers around any standard machine learning model (like LightGBM, Random Forest, or deep neural networks) to force them into an exploration/exploitation loop. They wrap any scikit-learn estimator that implements `partial_fit` and `predict`.

* **Contextual Epsilon-Greedy**: Uses a single base estimator. $1-\epsilon$ of the time it exploits the best prediction, and $\epsilon$ of the time it explores randomly.
* **Bootstrapped Thompson Sampling / UCB**: Maintains an ensemble of `n_bootstraps` models. Each model in the ensemble is updated with Poisson/Gamma-sampled weights (Online Bootstrapping) to simulate posterior uncertainty. At prediction time, TS picks one model randomly, while UCB takes the mean + standard deviation across the ensemble.

## 3. Key Hyperparameters

* `base_estimator`: Any scikit-learn estimator implementing `partial_fit` and `predict` (e.g., `SGDRegressor`, `LGBMRegressor`).
* `n_bootstraps` (Bootstrapped only): Number of models in the ensemble. More → better uncertainty estimate, higher memory/CPU cost. Typical range: 5–20.
* `epsilon` (EpsilonGreedy only): Exploration probability. `epsilon=0.1` means 10% random, 90% greedy.

## 3b. Random Forest Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rf_n_estimators` | `50` | Number of trees in the Random Forest ensemble |
| `rf_max_depth` | `6` | Maximum depth per tree. Higher → more complex, more overfitting risk |
| `rf_min_samples_leaf` | `1` | Minimum samples required to create a leaf node. Higher → smoother trees, less overfitting |
| `rf_max_obs` | `1000` | Maximum observations to store before FIFO eviction. Prevents unbounded memory growth |
| `rf_min_uncertainty` | `1e-6` | Lower bound for ensemble disagreement (uncertainty estimate). Prevents zero-division when trees agree too much |

## 3c. Bootstrap-Specific Parameters

| Policy | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| `BootstrappedUCB` | `bootstrap_method` | `"poisson"` | Poisson vs Gamma bootstrap resampling. Poisson more common for approximate bootstrap |
| `BootstrappedUCB` | `percentile` | `0.75` | UCB percentile: 0.75 means use 75th percentile of samples as the optimistic estimate |
| All Bootstrapped | `n_bootstraps` | `10` | Number of bootstrap models. Higher → better uncertainty, higher cost |

**Note:** `bootstrap_method` is selected internally based on policy type. For `BootstrappedUCB`, the percentile controls the trade-off between exploration and exploitation via the upper percentile of sampled scores.

## 4. PolicyType Mapping

| Policy | Class | Lesson | Use Case |
|--------|-------|--------|----------|
| `epsilon_greedy` | `EpsilonGreedyArmModel` | Not featured | Hard exploration cutoff |
| `bootstrapped_ts` | `BootstrappedTSArmModel` | Not featured | Ensemble + Bayesian uncertainty |
| `bootstrapped_ucb` | `BootstrappedUCBArmModel` | Not featured | Ensemble + confidence bounds |
| `random_forest_ucb` | Random Forest (sklearn) | Lesson 9 (Pricing) | Tree-based non-linear + UCB |
| `random_forest_ts` | Random Forest (sklearn) | Not featured | Tree-based non-linear + TS |

See [Policies Reference](../policies.md) for all 17 policies.

## 5. Example Usage

```python
from lightgbm import LGBMRegressor
from coba import ClusterBandit
from coba.types import PolicyType

bandit = ClusterBandit(
    arms=["A", "B", "C", "D"],
    n_features=7,
    policy=PolicyType.BOOTSTRAPPED_TS,
    n_clusters=5,
    base_estimator=LGBMRegressor(n_estimators=10, max_depth=3),
    n_bootstraps=10,  # maintains 10 distinct LightGBM models
)
```
