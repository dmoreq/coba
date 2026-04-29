# COBA: Generic Contextual Bandit Engine

**COBA** is a high-performance, domain-agnostic Contextual Bandit library for real-time decision making.

## Core Philosophy

1. **Domain Agnostic**: COBA operates purely on `numpy` arrays and floating-point math. It has zero knowledge of your domain. This forces a clean separation between the ML math and your business logic.
2. **KMeans Cluster Routing**: Instead of training one bandit for the entire context space, COBA uses a `ClusterRouter`. It groups incoming contexts into $K$ behaviorally distinct clusters and maintains independent bandits for each.
3. **High Performance**: Built on top of `numpy` with optimized Sherman-Morrison online updates for Ridge Regression, ensuring $O(d^2)$ updates without expensive matrix inversions in the critical path.

## Architecture

- `coba.bandit`: The main `ClusterBandit` facade — the recommended entry point.
- `coba.policies`: Core learning algorithms (`LinUCB`, `LinTS`, `UCB1`, `Thompson Sampling`, `Logistic`, `Bootstrapped`).
- `coba.router`: KMeans routing that maps context vectors to specialized per-cluster bandit models.
- `coba.evaluation`: Offline policy evaluation (Rejection Sampling, Doubly Robust, NCIS).
- `coba.offpolicy`: Inverse Propensity Scoring (IPS) utilities for bootstrapping from biased historical logs.

## Quick Start

```python
import numpy as np
from coba import ClusterBandit
from coba.types import PolicyType

# 1. Initialize Bandit
bandit = ClusterBandit(
    arms=["A", "B", "C", "D"],
    n_features=5,
    policy=PolicyType.LIN_UCB,
    n_clusters=3
)

# 2. Bootstrap from Historical Logs
bandit.fit_offline(
    contexts=np.random.randn(1000, 5),
    decisions=np.random.choice(["A", "B", "C", "D"], 1000),
    rewards=np.random.rand(1000),
    propensities=np.full(1000, 0.25)
)

# 3. Online Decision
context = np.array([0.5, -1.2, 0.3, 2.1, -0.8])
decision = bandit.decide(context)
print(f"Chosen arm: {decision.chosen_arm}")

# 4. Observe Reward and Update
bandit.update(context=context, arm=decision.chosen_arm, reward=0.85)
```

## Integrating with Your Domain

Build a **Domain Facade** that translates your domain objects into the raw `numpy` arrays COBA expects. COBA never needs to know what the context features or arm identifiers mean — that mapping lives entirely in your application layer.
