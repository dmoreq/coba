<h1 align="center">COBA</h1>
<p align="center">
  <em>A High-Performance, Domain-Agnostic Contextual Bandit Engine for Dynamic Pricing</em>
</p>

---

**COBA** (COntextual BANdit engine) is a lightweight, high-performance reinforcement learning library designed to handle real-time decision making and dynamic pricing using Multi-Armed Bandits.

By treating the context as a raw multi-dimensional feature vector, COBA strictly separates the machine learning math from your business logic, allowing it to seamlessly integrate into any existing domain architecture (like Geofencing, H3 Cells, or User Targeting).

📖 **[Đọc tài liệu Tiếng Việt tại đây (Vietnamese Documentation)](docs/vi/index.md)**

## ⚡ Key Features

* **Domain Agnostic**: Operates entirely on raw `numpy` arrays. No dependencies on specific databases, ORMs, or business domains.
* **Smart Cluster Routing**: Uses `MiniBatchKMeans` to route incoming contexts into distinct market regimes ($K$ clusters), maintaining an independent bandit model for each regime to accelerate learning.
* **$O(d^2)$ Online Updates**: Powered by the Sherman-Morrison formula for online Ridge Regression. Avoids expensive matrix inversions ($O(d^3)$) during real-time updates.
* **Multiple Policies**: Supports contextual (`LinUCB`, `LinTS`, `LogisticUCB`, `LogisticTS`), non-linear (`BootstrappedTS`, `BootstrappedUCB`, `EpsilonGreedy`), and context-free (`UCB1`, `Thompson Sampling`) algorithms.
* **Non-stationary Support**: All linear policies accept a `gamma` discount factor to forget old data and adapt to changing market trends.
* **Offline Evaluation**: Validate your policies on historical log data using `Rejection Sampling`, `Doubly Robust`, or `NCIS` metrics before running A/B tests.

## 🗂 Project Structure

```text
coba/
├── cluster_bandit.py   # Main Facade and entrypoint
├── policies/           # Core Bandit algorithms (LinUCB, LinTS, etc.)
├── routers/            # KMeans cluster routing
├── evaluation/         # Offline policy evaluation metrics
├── offpolicy/          # Inverse Propensity Scoring (IPS) utils
└── schemas.py          # Pydantic schemas for I/O
```

## 🚀 Quick Start

To use COBA, simply import the `ClusterBandit` facade into your application.

```python
import numpy as np
from coba.cluster_bandit import ClusterBandit
from coba.types import PolicyType

# 1. Initialize Bandit Engine
bandit = ClusterBandit(
    arms=[1.0, 1.1, 1.2, 1.5],   # Example: Price Multipliers
    n_features=7,                # Context dimension
    policy=PolicyType.LIN_TS,    # Linear Thompson Sampling
    n_clusters=3                 # Number of distinct market regimes
)

# 2. Bootstrap from Historical Logs (IPS correction)
bandit.fit_from_logs(
    contexts=np.random.randn(1000, 7),
    decisions=np.random.choice([1.0, 1.1, 1.2, 1.5], 1000),
    rewards=np.random.rand(1000),
    propensities=np.full(1000, 0.25)
)

# 3. Online Decision Making
context_vector = np.array([10.5, 2.0, 50, 10, 8, 2, 5.25])
decision = bandit.decide(context_vector)

print(f"Algorithm chose arm: {decision.chosen_arm}")

# 4. Online Feedback Update
bandit.update(
    context=context_vector,
    arm=decision.chosen_arm,
    reward=0.85
)
```

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

* **English**:
  * [Overview & Architecture](docs/index.md)
  * [Policy References](docs/policies.md)
  * [Offline Evaluation](docs/evaluation.md)
* **Tiếng Việt**:
  * [Tổng quan hệ thống](docs/vi/index.md)
  * [Các thuật toán MAB](docs/vi/policies.md)
  * [Đánh giá ngoại tuyến (Offline Evaluation)](docs/vi/evaluation.md)

## 🧪 Testing & Development

COBA uses a `Makefile` to standardize code quality workflows:

```bash
make lint      # Run Ruff linter
make format    # Run Black formatter
make test      # Run pytest (fast, no coverage)
make coverage  # Run pytest with 90% coverage threshold
```

You can also run pytest directly:

```bash
python -m pytest tests/ -v
```
*(Current coverage: >95%)*

For coverage reports:

```bash
python -m pytest -q --cov=. --cov-report=term
```

For `bandit_by_geo` integration tests from the `experiments/` directory:

```bash
PYTHONPATH=$(pwd) python -m pytest -q -p no:asyncio bandit_by_geo/tests
```

## 🛡 Production Readiness Notes

- **Input validation**: `ClusterBandit` now validates context shape, feature count, finite values, and propensity bounds before updates.
- **Off-policy safety**: `use_dr=True` requires `reward_estimates`. If fallback behavior is explicitly needed in incremental updates, use:
  `IPSConfig(use_dr=True, allow_ips_fallback_when_dr_missing=True)`.
- **Backward compatibility**: `coba.persistence` keeps `save_model`/`load_model` aliases for older code while the canonical API remains `save_bandit`/`load_bandit`.
- **Monitoring hooks**: `ClusterBandit.get_stats()` exposes per-arm pull counts and mean rewards, and offline evaluation helpers are available via `evaluate_rejection_sampling`, `evaluate_doubly_robust`, and `evaluate_ncis`.
- **Logging**: COBA uses structured `loguru` logs for cold start decisions, persistence actions, and batch updates; route these logs to your centralized collector in production.
