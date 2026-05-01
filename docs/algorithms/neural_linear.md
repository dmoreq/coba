# NeuralLinear (Neural Network + Linear Thompson Sampling)

## 1. Overview

* **Type**: Non-linear (MLP) + Bayesian linear (LinTS)
* **Class**: `NeuralLinearArmModel` (`policies/neural_linear.py`)
* **Best for**: Non-linear reward surfaces where you still want calibrated Bayesian exploration — without a GPU.

## 2. How It Works

NeuralLinear is a two-tier architecture: a shared MLP extracts non-linear embeddings, and each arm runs a per-arm LinTS head on those embeddings.

### Tier 1 — Shared MLP Backbone (`NeuralLinearBackbone`)

One backbone is maintained **per cluster** (not per arm). It is an sklearn `MLPRegressor` with architecture:

$$\text{layers} = \text{hidden\_sizes} + (\text{embedding\_dim},)$$

The backbone maintains a **shared replay buffer** (FIFO, max `10 000` entries) across all arms in the cluster, storing tuples `(x, arm, reward, weight)`.

Every `retrain_freq` total updates, the backbone refits its MLP on the full buffer. The **penultimate layer activations** (before the output regression head, using ReLU) are used as the embedding:

$$\phi(x) = \text{ReLU}(W_{L-1} \cdots \text{ReLU}(W_1 x + b_1) \cdots + b_{L-1}) \in \mathbb{R}^{d_{\text{emb}}}$$

### Tier 2 — Per-Arm LinTS Head

Each arm maintains a `LinTSArmModel` on the embedding space $\mathbb{R}^{d_{\text{emb}}}$. After a backbone retrain, **all per-arm LinTS heads are rebuilt from scratch** by replaying their buffered observations through the new embedding — incremental updates are discarded.

Between retrains, incoming observations update the LinTS head incrementally via Sherman-Morrison.

### Cold Start

Until the backbone is fitted, `score()` returns `float("inf")` so every arm is explored at least once before learning begins.

### Complexity

| Operation | Cost |
|-----------|------|
| `update()` | O(1) amortized (buffer append); O(n·d_emb²) per retrain trigger |
| `score()` | O(d_emb²) — LinTS sample on embedding |
| Memory | O(buffer\_maxlen × n\_features) for the buffer |

## 3. Key Hyperparameters

* `neural_embedding_dim`: Dimensionality of the penultimate MLP layer — this is the LinTS input size (default `16`). Larger → richer representation but slower LinTS updates.
* `neural_hidden_sizes`: Tuple of hidden layer widths (default `(64, 32)`). An output layer of width `embedding_dim` is appended automatically. Example: `(64, 32)` produces layers 64 → 32 → 16 (with `embedding_dim=16`).
* `neural_retrain_freq`: Total arm updates between backbone retrains (default `200`). Lower → backbone adapts faster but uses more CPU. Higher → faster serving but backbone lags recent reward distributions.
* `v_sq`: Posterior variance multiplier for the LinTS head (default `1.0`). Higher → more exploration.
* `l2_lambda`: L2 regularization for both the MLP (sklearn `alpha`) and per-arm LinTS (default `1.0`).
* `gamma`: Discount factor for non-stationarity (default `1.0`). Applied to the per-arm LinTS heads.

## 4. Example

```python
from coba import ClusterBandit
from coba.types import PolicyType

bandit = ClusterBandit(
    arms=["A", "B", "C"],
    n_features=20,
    policy=PolicyType.NEURAL_LINEAR,
    neural_embedding_dim=16,       # penultimate layer size
    neural_hidden_sizes=(64, 32),  # actual layers: 64 → 32 → 16
    neural_retrain_freq=100,       # retrain backbone every 100 updates
    v_sq=1.0,
    l2_lambda=1.0,
    n_clusters=3,
)

import numpy as np
context = np.random.randn(20)

# First ~100 updates: backbone unfitted, score() returns inf → pure exploration
bandit.fit(contexts, arms_chosen, rewards)

decision = bandit.decide(context)
print(decision.chosen_arm, decision.score)
```

Standalone backbone usage:

```python
from coba.policies.neural_linear import NeuralLinearBackbone, NeuralLinearArmModel

backbone = NeuralLinearBackbone(
    n_features=20,
    embedding_dim=16,
    hidden_sizes=(64, 32),
    retrain_freq=100,
)

arm_a = NeuralLinearArmModel(arm="A", backbone=backbone, v_sq=1.0)
arm_b = NeuralLinearArmModel(arm="B", backbone=backbone, v_sq=1.0)

x = np.random.randn(20)
arm_a.update(x, reward=0.8)  # feeds backbone buffer; may trigger retrain

embedding = backbone.get_embedding(x)  # None until backbone is fitted
```

## 5. When to Use

| Scenario | Recommendation |
|----------|----------------|
| Non-linear rewards, Bayesian exploration, no GPU | **NeuralLinear** |
| Linear rewards, high throughput (> 10k/s) | LinUCB or LinTS |
| Non-linear, sklearn flexibility, no Bayesian requirement | BootstrappedTS |
| Very low volume (< 1k obs/arm), complex non-linear | GP-UCB |
| Large volume + GPU available | External deep RL library |

**Note**: NeuralLinear retrains the backbone periodically — each retrain is an O(n) MLP fit on the full buffer. For very high-throughput systems, set `neural_retrain_freq` high (≥ 500) or prefer LinUCB/LinTS.

## 6. References

Riquelme, C., Tucker, G., & Snoek, J. (2018). *Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling*. ICLR 2018.
