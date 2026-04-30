# LinTS (Linear Thompson Sampling)

## 1. Overview

* **Type**: Stochastic / Bayesian
* **Class**: `LinTSArmModel` (`policies/lin_ts.py`)
* **Best for**: Production systems. Often outperforms LinUCB empirically because stochastic exploration handles delayed feedback and batch updates better.

## 2. How It Works

LinTS maintains a Bayesian posterior distribution over the true reward weights. At each call, it draws a random coefficient vector $\tilde{\beta}$ from the Gaussian posterior:

$$\tilde{\beta} \sim \mathcal{N}(\hat{\beta}, v^2 A^{-1}), \quad \text{score}(x) = x^\top \tilde{\beta}$$

Because the score depends on a random sample, exploration is implicit: arms with high uncertainty have wider distributions, so they naturally receive high scores more often. As data accumulates, the posterior narrows and the algorithm exploits more.

This is particularly useful when feedback is delayed (e.g., batch updates): unlike LinUCB which would greedily over-select one arm during a delay, LinTS distributes selections across arms by sampling different $\tilde{\beta}$ each call.

## 3. Key Hyperparameters

* `v_sq`: Posterior variance multiplier. Higher → wider distribution → more exploration.
* `l2_lambda`: L2 regularization (default 1.0).
* `gamma`: Discount factor for non-stationarity (default 1.0). Set `< 1.0` (e.g., 0.99) to forget old data.

## 4. Example

```python
import numpy as np
from coba.policies.lin_ts import LinTSArmModel

model = LinTSArmModel(arm="variant_A", n_features=3, v_sq=1.0, gamma=0.99)

context = np.array([0.8, 0.2, 0.9])

# Each call returns a different score because of stochastic sampling —
# this is the exploration mechanism.
score_1 = model.score(context)
score_2 = model.score(context)
print(f"Sample 1: {score_1:.4f} | Sample 2: {score_2:.4f}")

# Update with observed reward
model.update(context, reward=1.0)
# As more data arrives, the two samples above will converge closer together.
```
