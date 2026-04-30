<h1 align="center">COBA</h1>
<p align="center">
  <em>A High-Performance, Domain-Agnostic Contextual Bandit Engine</em>
</p>

---

**COBA** (COntextual BAndit engine) is a lightweight, high-performance reinforcement learning library for real-time decision making using Multi-Armed Bandits.

By treating the context as a raw multi-dimensional feature vector, COBA strictly separates the machine learning math from your business logic, allowing it to seamlessly integrate into any existing domain architecture.

📖 **[Đọc tài liệu Tiếng Việt tại đây (Vietnamese Documentation)](docs/vi/index.md)**

## ⚡ Key Features

* **Domain Agnostic**: Operates entirely on raw `numpy` arrays. No dependencies on specific databases, ORMs, or business domains.
* **Smart Cluster Routing**: Uses `MiniBatchKMeans` to route incoming contexts into $K$ distinct clusters, maintaining an independent bandit model per cluster to accelerate learning.
* **$O(d^2)$ Online Updates**: Powered by the Sherman-Morrison formula for online Ridge Regression. Avoids expensive matrix inversions ($O(d^3)$) during real-time updates.
* **Multiple Policies**: Supports contextual (`LinUCB`, `LinTS`, `LogisticUCB`, `LogisticTS`), non-linear (`BootstrappedTS`, `BootstrappedUCB`, `EpsilonGreedy`), and context-free (`UCB1`, `Thompson Sampling`) algorithms.
* **Non-stationary Support**: All linear policies accept a `gamma` discount factor to forget old data and adapt to distribution shifts.
* **Offline Evaluation**: Validate policies on historical log data using `Rejection Sampling`, `Doubly Robust`, or `NCIS` metrics before running A/B tests.

## 🗂 Project Structure

```text
src/coba/
├── bandit.py       # Main facade and entry point
├── router.py       # KMeans cluster routing
├── offpolicy.py    # Inverse Propensity Scoring (IPS) utilities
├── evaluation.py   # Offline policy evaluation metrics
├── schemas.py      # Pydantic schemas for I/O
├── persistence.py  # joblib-based save/load
└── policies/       # Core bandit algorithms (LinUCB, LinTS, etc.)
```

## 🚀 Quick Start

```python
import numpy as np
from coba import ClusterBandit
from coba.types import PolicyType

# 1. Initialize the bandit
bandit = ClusterBandit(
    arms=["A", "B", "C", "D"],
    n_features=5,
    policy=PolicyType.LIN_TS,
    n_clusters=3
)

# 2. Bootstrap from historical logs (with IPS correction)
bandit.fit_offline(
    contexts=np.random.randn(1000, 5),
    decisions=np.random.choice(["A", "B", "C", "D"], 1000),
    rewards=np.random.rand(1000),
    propensities=np.full(1000, 0.25)
)

# 3. Online decision making
ctx = np.array([0.5, -1.2, 0.3, 2.1, -0.8])
decision = bandit.decide(ctx)
print(f"Chosen arm: {decision.chosen_arm}")

# 4. Online feedback update
bandit.update(context=ctx, arm=decision.chosen_arm, reward=0.85)
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

## 💡 Examples

See the [`examples/`](examples/) directory for runnable demonstrations of all features:

| File | What it shows |
|---|---|
| `examples/01_quickstart.py` | Basic decide → update loop |
| `examples/02_policy_types.py` | Every `PolicyType` in action |
| `examples/03_offline_bootstrap.py` | `fit_offline` with IPS & DR correction |
| `examples/04_arm_management.py` | Add/remove arms with warm start |
| `examples/05_evaluation.py` | Rejection sampling, DR, NCIS metrics |
| `examples/06_persistence.py` | Save and load a bandit |
| `examples/07_batch_update.py` | Batch and streaming log-replay |
| `examples/08_monitoring.py` | Per-arm stats, scores, cluster assignments |

## 🧪 Testing & Development

```bash
make lint      # Run Ruff linter
make format    # Run Black formatter
make test      # Run pytest (fast, no coverage)
make coverage  # Run pytest with 90% coverage threshold
```

Run a single test file:
```bash
pytest tests/test_bandit.py -v
```

## 🛡 Production Readiness Notes

- **Input validation**: `ClusterBandit` validates context shape, feature count, finite values, and propensity bounds before updates.
- **Off-policy safety**: `use_dr=True` requires `reward_estimates`. For explicit IPS fallback in incremental updates, use `IPSConfig(use_dr=True, allow_ips_fallback_when_dr_missing=True)`.
- **Backward compatibility**: `coba.persistence` keeps `save_model`/`load_model` aliases while the canonical API is `save_bandit`/`load_bandit`.
- **Monitoring**: `ClusterBandit.get_stats()` exposes per-arm pull counts and mean rewards.
- **Logging**: COBA uses structured `loguru` logs for cold-start decisions, persistence actions, and batch updates.
