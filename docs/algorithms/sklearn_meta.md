# Meta-Heuristics (Scikit-learn Wrappers)

## 1. Overview

* **Type**: Non-linear / Meta-algorithm
* **Classes**: `EpsilonGreedyArmModel`, `BootstrappedTSArmModel`, `BootstrappedUCBArmModel` (`policies/sklearn_models.py`)
* **Best for**: Complex non-linear feature interactions (e.g., categorical features, time-of-day interactions) where a simple linear model underfits.

## 2. How It Works

These policies act as wrappers around any standard machine learning model (like LightGBM, Random Forest, or deep neural networks) to force them into an exploration/exploitation loop. They wrap any scikit-learn estimator that implements `partial_fit` and `predict`.

* **Contextual Epsilon-Greedy**: Uses a single base estimator. $1-\epsilon$ of the time it exploits the best prediction, and $\epsilon$ of the time it explores randomly.
* **Bootstrapped Thompson Sampling / UCB**: Maintains an ensemble of `n_bootstraps` models. Each model in the ensemble is updated with Poisson/Gamma-sampled weights (Online Bootstrapping) to simulate posterior uncertainty. At prediction time, TS picks one model randomly, while UCB takes the mean + standard deviation across the ensemble.

## 3. Example Usage

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
