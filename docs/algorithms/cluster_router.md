# Cluster Routing

## 1. The Intuition

In dynamic pricing for ride-hailing or delivery, a high-demand downtown area during rush hour behaves fundamentally differently from a quiet suburb at midnight. A single linear model struggles to capture these contrasting behaviors.

COBA handles non-linear and non-stationary environments using a **Cluster Router**. Rather than training a single, massive bandit model for all possible contexts, the `ClusterRouter` splits the market into distinct "regimes" or clusters and manages an independent bandit for each.

## 2. How It Works

By utilizing KMeans clustering on the context vectors:
1. **Specialization**: Each cluster learns a specialized model tailored to its specific market conditions.
2. **Efficiency**: Training $K$ small linear models is significantly faster and more stable than training one highly complex non-linear model.
3. **Stability**: We use `MiniBatchKMeans` to support incremental (online) learning without completely restructuring the clusters on every update.

## 3. Dynamic Arm Management

The `ClusterRouter` natively supports adding and removing arms on the fly, which is essential for dynamic pricing systems where you might introduce a new price tier.

* **Warm Start**: When adding a new arm, you can copy the trained parameters of an existing arm. This prevents the new arm from starting completely randomly (cold start) and destroying user experience.
* **Atomic Updates**: Adding or removing an arm updates all underlying cluster bandits atomically.

## 4. Example Usage

Here is a step-by-step example of how to use the `ClusterRouter`.

```python
import numpy as np
from coba.routers.cluster_router import ClusterRouter
from coba.types import PolicyType

# 1. Initialize the router
# We start with 3 pricing arms and 5 clusters
router = ClusterRouter(
    arms=[1.0, 1.2, 1.5],
    n_clusters=5,
    policy=PolicyType.LIN_UCB,
    n_features=4,
    use_minibatch=True, # Recommended for online production systems
    scale_contexts=True # Standardizes features before clustering
)

# 2. Offline Training (Batch Fit)
# In reality, this data comes from your historical logs
n_samples = 1000
contexts = np.random.randn(n_samples, 4)
decisions = np.random.choice([1.0, 1.2, 1.5], size=n_samples)
rewards = np.random.rand(n_samples)

router.fit(contexts, decisions, rewards)
print(f"Is fitted: {router.is_fitted}") # True

# 3. Online Prediction
# A new request comes in
new_context = np.array([2.5, 0.5, 1.0, -0.2])
chosen_arm = router.predict(new_context)
print(f"Chosen price multiplier: {chosen_arm}")

# 4. Online Update
# The user accepted the price (reward = 1.0)
router.update(new_context, chosen_arm, reward=1.0)

# 5. Dynamic Arm Management (Warm Start)
# We want to introduce a new 1.3x multiplier.
# We "warm start" it using the learned weights of the 1.2x arm to avoid random behavior.
router.add_arm(arm=1.3, warm_start_from=1.2)

# Verify the new arm exists
scores = router.score_all(new_context)
print(f"Scores for all arms: {scores}")
# Output will now include 1.3
```
