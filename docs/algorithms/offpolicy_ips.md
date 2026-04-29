# Off-Policy Learning: IPS & Doubly-Robust

## 1. The Intuition

When bootstrapping a new bandit from historical data, that data was collected by a prior policy (e.g., a rule-based system or a random A/B test). This is known as **Off-Policy Learning**.

The problem is that historical data is **biased**: if the logging policy preferred arm "A", the logs are saturated with "A" observations while under-representing others. Naively training on this data inherits the bias.

## 2. Inverse Propensity Scoring (IPS)

COBA corrects for this by reweighting each observation by the inverse probability (the **propensity**) that the logging policy chose that arm:

* **High propensity (common action)** → low weight. Already well-represented; down-weight its influence.
* **Low propensity (rare action)** → high weight. Under-represented; up-weight the few observed examples.

### Doubly-Robust (DR) Correction

Pure IPS can suffer from high variance when propensities are very small (producing extreme weights). COBA also implements **Doubly-Robust** correction, which combines IPS with a reward prediction model to reduce variance while remaining unbiased.

### Runtime Contract

- `fit_from_logs(..., use_dr=True)` requires `reward_estimates`.
- `update_from_logs(..., use_dr=True)` also requires `reward_estimates` by default.
- To allow fallback to IPS when estimates are missing in incremental updates:
  `IPSConfig(use_dr=True, allow_ips_fallback_when_dr_missing=True)`.

## 3. Example Usage

```python
import numpy as np
from coba.offpolicy import DoublyRobustUpdater, IPSConfig
from coba.router import ClusterRouter
from coba.types import PolicyType

# 1. Initialize an empty router
router = ClusterRouter(
    arms=["A", "B", "C"],
    n_clusters=3,
    policy=PolicyType.LIN_UCB,
    n_features=4
)

# 2. Simulate biased historical logs
n_samples = 1000
contexts  = np.random.randn(n_samples, 4)
# Logging policy heavily favored arm "A"
decisions    = np.random.choice(["A", "B", "C"], size=n_samples, p=[0.7, 0.2, 0.1])
rewards      = np.random.rand(n_samples)
propensities = np.array([0.7 if d == "A" else 0.2 if d == "B" else 0.1 for d in decisions])

# 3. Configure IPS correction
config  = IPSConfig(clip_min=1e-4, clip_max=10.0, use_dr=False)
updater = DoublyRobustUpdater(router, config)

# 4. Bootstrap from logs (de-biases and fits the router)
updater.fit_from_logs(
    contexts=contexts,
    decisions=decisions,
    rewards=rewards,
    propensities=propensities,
)
print(f"Router fitted: {router.is_fitted}")

# 5. Incremental update from a new batch (does NOT refit KMeans clusters)
new_ctx  = np.random.randn(100, 4)
new_dec  = np.random.choice(["A", "B", "C"], size=100)
new_rew  = np.random.rand(100)
new_prop = np.full(100, 1 / 3)

updater.update_from_logs(
    contexts=new_ctx,
    decisions=new_dec,
    rewards=new_rew,
    propensities=new_prop,
)
```
