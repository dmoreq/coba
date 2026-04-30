# GP-UCB (Gaussian Process Upper Confidence Bound)

## 1. Overview

* **Type**: Bayesian / Non-parametric
* **Class**: `GPUCBArmModel` (`policies/gp_ucb.py`)
* **Best for**: Low-volume decisions with expensive observations, complex non-linear reward surfaces, or when you have a strong prior on reward smoothness.

## 2. How It Works

GP-UCB maintains a Gaussian Process posterior over the reward function. At each step, the score combines the posterior mean (exploitation) with the posterior standard deviation scaled by `beta` (exploration):

$$\text{score}(x) = \underbrace{\mu(x)}_{\text{exploit}} + \underbrace{\beta \cdot \sigma(x)}_{\text{explore}}$$

Where the posterior is computed with an **RBF (Radial Basis Function)** kernel:

$$k(x, x') = \exp\!\left(-\frac{\|x - x'\|^2}{2 \ell^2}\right)$$

The posterior mean and variance are:

$$\mu(x) = k(x, X)^\top (K + \sigma_n^2 I)^{-1} y$$
$$\sigma^2(x) = k(x,x) - k(x, X)^\top (K + \sigma_n^2 I)^{-1} k(X, x)$$

The matrix $(K + \sigma_n^2 I)$ is factored via **Cholesky decomposition** at each `score()` call (cached, rebuilt lazily).

### Complexity

| Operation | Cost |
|-----------|------|
| `update()` | O(1) — appends observation |
| `score()` | O(n²) — Cholesky + triangular solve |
| Memory | O(n²) — stores the full kernel matrix |

This makes GP-UCB best for **low-to-medium volume** use cases. For high-throughput serving, prefer LinUCB or LinTS.

## 3. Key Hyperparameters

* `gp_beta`: UCB exploration coefficient (default `2.0`). Higher → more exploration. Values 1–5 are typical.
* `gp_length_scale`: RBF kernel bandwidth `ℓ` (default `1.0`). Controls how quickly reward similarity decays with distance. Smaller `ℓ` → arms treat nearby contexts as dissimilar (high curvature). Larger `ℓ` → smoother generalization across contexts.
* `gp_noise_var`: Observation noise variance `σ_n²` (default `0.1`). Acts like L2 regularization — higher values make the GP interpolate less aggressively.
* `gp_max_obs`: Maximum stored observations per arm (default `500`). When exceeded, oldest observations are dropped (FIFO) to keep inference tractable.

## 4. Example

```python
import numpy as np
from coba.policies.gp_ucb import GPUCBArmModel

# Per-arm model (standalone use)
model = GPUCBArmModel(arm="variant_A", beta=2.0, length_scale=1.0, noise_var=0.1)

x = np.array([0.5, -0.3, 1.2])
model.update(x, reward=0.8)
print(model.score(np.array([0.6, -0.2, 1.1])))  # posterior mean + beta * std
```

```python
from coba import ClusterBandit
from coba.types import PolicyType

# Via ClusterBandit facade
bandit = ClusterBandit(
    arms=["A", "B", "C"],
    n_features=5,
    policy=PolicyType.GP_UCB,
    n_clusters=3,
    gp_beta=2.0,
    gp_length_scale=1.0,
    gp_noise_var=0.1,
    gp_max_obs=200,
)
```

## 5. When to Use GP-UCB

| Scenario | Recommendation |
|----------|----------------|
| < 1k decisions per arm, reward surface is non-linear | **GP-UCB** |
| High-throughput (> 10k/s), linear reward structure | LinUCB or LinTS |
| Non-linear, high-volume, budget for sklearn overhead | BootstrappedTS |

## 6. References

Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2010). *Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design*. ICML 2010.
