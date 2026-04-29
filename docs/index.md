# COBA: Generic Contextual Bandit Engine

**COBA** is a high-performance, domain-agnostic Contextual Bandit library designed for real-time dynamic pricing and decision making. 

## Core Philosophy
1. **Domain Agnostic**: COBA operates purely on `numpy` arrays and floating-point math. It has zero knowledge of your domain (e.g., H3 indexes, geofencing, or users). This forces a clean separation of concerns.
2. **KMeans Cluster Routing**: Instead of training one massive bandit for the entire context space, COBA uses a `ClusterRouter`. It groups incoming contexts into $K$ market regimes (e.g., "Rush hour CBD", "Quiet Suburb") and maintains independent bandits for each regime.
3. **High Performance**: Built on top of `numpy` with optimized Sherman-Morrison online updates for Ridge Regression, ensuring $O(d^2)$ updates without expensive matrix inversions in the critical path.

## Architecture

The library is organized into the following modules:

- `coba.cluster_bandit`: The main `ClusterBandit` Facade. This is the entrypoint for all domain consumers.
- `coba.policies`: Core machine learning algorithms (`LinUCB`, `LinTS`, `UCB1`, `Thompson Sampling`).
- `coba.routers`: Routing logic that maps high-dimensional context vectors to specific specialized bandit models using `MiniBatchKMeans`.
- `coba.evaluation`: Offline policy evaluation methods (Rejection Sampling, Doubly Robust, NCIS).
- `coba.offpolicy`: Inverse Propensity Scoring (IPS) utilities for bootstrapping models from biased historical logs.

## Quick Start

```python
import numpy as np
from coba.cluster_bandit import ClusterBandit
from coba.types import PolicyType

# 1. Initialize Bandit
bandit = ClusterBandit(
    arms=[1.0, 1.1, 1.2, 1.5],
    n_features=7,
    policy=PolicyType.LIN_UCB,
    n_clusters=5
)

# 2. Bootstrap from Historical Logs (Offline)
bandit.fit_from_logs(
    contexts=np.random.randn(1000, 7),
    decisions=np.random.choice([1.0, 1.1, 1.2, 1.5], 1000),
    rewards=np.random.rand(1000),
    propensities=np.full(1000, 0.25) # Probability logging policy chose the arm
)

# 3. Serve in Production (Online)
context_vector = np.array([10.5, 2.0, 50, 10, 8, 2, 5.25])
decision = bandit.decide(context_vector)
print(f"Chosen Arm: {decision.chosen_arm}")

# 4. Receive Feedback and Update
bandit.update(
    context=context_vector,
    arm=decision.chosen_arm,
    reward=0.85
)
```

## Integrating with your Domain
To use COBA in a microservice (like FastAPI), you should build a **Domain Facade** that translates your Pydantic schemas (e.g., `H3PricingContext`) into the raw `numpy` arrays expected by COBA. See `bandit_by_location/models/bandit.py` for a reference implementation.
