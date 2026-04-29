# Cluster Routing

## 1. The Intuition

Context spaces are rarely uniform. Different regions of the feature space may exhibit very different reward patterns — a single linear model that must cover the entire space will either overfit one region or underfit another.

COBA handles this via a **Cluster Router**. Rather than training one massive bandit for all possible contexts, the `ClusterRouter` partitions the context space into $K$ distinct clusters and maintains an independent bandit per cluster.

## 2. How It Works

Using KMeans clustering on context vectors:
1. **Specialization**: Each cluster learns a model tailored to its local context region.
2. **Efficiency**: $K$ small linear models are faster and more numerically stable than one complex non-linear model.
3. **Online stability**: `MiniBatchKMeans` supports incremental updates without restructuring all clusters on every observation.

## 3. Dynamic Arm Management

The `ClusterRouter` supports adding and removing arms at runtime across all clusters atomically.

* **Warm Start**: Copy trained parameters from an existing arm into the new one, avoiding random cold-start behavior.
* **Atomic Updates**: Add/remove propagates to every cluster bandit in one operation.

## 4. Example Usage

```python
import numpy as np
from coba.router import ClusterRouter
from coba.types import PolicyType

# 1. Initialize the router
router = ClusterRouter(
    arms=["A", "B", "C"],
    n_clusters=5,
    policy=PolicyType.LIN_UCB,
    n_features=4,
    use_minibatch=True,   # recommended for online systems
    scale_contexts=True   # standardizes features before clustering
)

# 2. Batch fit from historical data
n_samples = 1000
contexts  = np.random.randn(n_samples, 4)
decisions = np.random.choice(["A", "B", "C"], size=n_samples)
rewards   = np.random.rand(n_samples)

router.fit(contexts, decisions, rewards)
print(f"Fitted: {router.is_fitted}")  # True

# 3. Online prediction
ctx = np.array([2.5, 0.5, 1.0, -0.2])
chosen = router.predict(ctx)
print(f"Chosen arm: {chosen}")

# 4. Online update
router.update(ctx, chosen, reward=0.9)

# 5. Add a new arm with warm start from "B"
router.add_arm(arm="D", warm_start_from="B")

scores = router.score_all(ctx)
print(f"All arm scores: {scores}")  # now includes "D"
```
