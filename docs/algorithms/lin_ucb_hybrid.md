# LinUCB-Hybrid (Hybrid Linear Upper Confidence Bound)

## 1. Overview

* **Type**: Deterministic / Optimism in the Face of Uncertainty
* **Class**: `LinUCBHybridArmModel` (`policies/lin_ucb_hybrid.py`)
* **Best for**: Scenarios where the context contains both **shared features** (user demographics, session signals) that generalize across all arms and **arm-specific features** (item embeddings, content attributes) unique to each arm.

## 2. How It Works

The full context vector `x` (length `n_features = n_shared + n_arm`) is split into two parts:

* `z = x[:n_shared]` — shared features, common to all arms (e.g. user age, time-of-day)
* `x_arm = x[n_shared:]` — arm-specific features (e.g. item category embedding)

**One `SharedRidge` instance** is maintained per cluster and updated on **every** arm pull regardless of which arm was chosen. Each arm also maintains its own private `RidgeRegression` trained only on that arm's observations.

The score combines exploitation and an UCB exploration bonus from both components:

$$\text{score}(z, x_{\text{arm}}) = \underbrace{z^\top \hat{\beta}_{\text{shared}} + x_{\text{arm}}^\top \hat{\theta}_{\text{arm}}}_{\text{exploit}} + \underbrace{\alpha \sqrt{z^\top A_0^{-1} z + x_{\text{arm}}^\top A_{\text{arm}}^{-1} x_{\text{arm}}}}_{\text{explore}}$$

**Key property**: because all arm pulls update `β_shared`, shared features converge much faster than arm-specific features — you get cross-arm transfer learning for free on the shared dimensions.

> This is an approximation of the full hybrid UCB from Li et al. (the exact form includes a cross-covariance term). The approximation retains the main practical benefit while keeping the implementation O(d²).

### Complexity

| Operation | Cost |
|-----------|------|
| `update()` | O(d_shared²) + O(d_arm²) — two Sherman-Morrison updates |
| `score()` | O(d_shared²) + O(d_arm²) — two quadratic forms |
| Memory | O(d_shared²) shared + O(d_arm² × n_arms) per-arm |

## 3. Key Hyperparameters

* `n_shared_features` (set on `ClusterBandit`): number of shared context dimensions — the **first** `n_shared_features` elements of each context vector. Must be ≥ 0. Default `0` (pure per-arm, same as standard LinUCB).
* `alpha`: UCB exploration width (default `1.0`). Higher → wider confidence intervals → more exploration. Typical range: 0.5–2.0.
* `l2_lambda`: L2 regularization for the per-arm ridge (default `1.0`).
* `gamma`: Discount factor for non-stationarity (default `1.0`). Set `< 1.0` (e.g. `0.99`) to forget old observations and track distribution shifts.

## 4. Example

```python
import numpy as np
from coba import ClusterBandit
from coba.types import PolicyType

# Context layout: [user_age, session_length, time_of_day, city_id,  ← shared (4 dims)
#                  item_category, item_price, item_popularity, ...]   ← arm-specific (6 dims)

bandit = ClusterBandit(
    arms=["article_A", "article_B", "article_C"],
    n_features=10,            # total context length
    n_shared_features=4,      # first 4 dims learned jointly across all arms
    policy=PolicyType.LIN_UCB_HYBRID,
    alpha=1.0,
    l2_lambda=1.0,
    n_clusters=3,
)

context = np.random.randn(10)

# Before fitting: returns first arm (cold start)
decision = bandit.decide(context)

# After collecting data:
bandit.fit(contexts, arms_chosen, rewards)
decision = bandit.decide(context)
print(decision.chosen_arm, decision.score)
```

Standalone per-arm usage:

```python
from coba.policies.lin_ucb_hybrid import LinUCBHybridArmModel
from coba.policies.ridge import RidgeRegression

# Shared ridge — one per cluster, passed to all arms
shared = RidgeRegression(n_features=4, l2_lambda=1.0)

arm_a = LinUCBHybridArmModel(arm="A", n_shared=4, n_arm=6, shared_ridge=shared, alpha=1.0)
arm_b = LinUCBHybridArmModel(arm="B", n_shared=4, n_arm=6, shared_ridge=shared, alpha=1.0)

x = np.random.randn(10)
arm_a.update(x, reward=1.0)  # updates shared ridge AND arm_a's private ridge
print(arm_b.score(x))        # arm_b benefits from shared ridge update immediately
```

## 5. When to Use

| Scenario | Recommendation |
|----------|----------------|
| Context has clear shared dims (user) + arm-specific dims (item) | **LinUCB-Hybrid** |
| All context is arm-specific, no cross-arm transfer expected | LinUCB |
| Need stochastic exploration or delayed-feedback robustness | LinTS |
| Non-linear reward surface, medium volume | NeuralLinear or BootstrappedTS |
| Very low volume, complex non-linear surface | GP-UCB |

## 6. References

Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). *A Contextual-Bandit Approach to Personalized News Article Recommendation*. WWW 2010.
