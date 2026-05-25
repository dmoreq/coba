# COBA Documentation

COBA is a Python library for contextual bandits, offline policy evaluation, drift detection, and continuous-action experimentation.

## Getting Started

- [Architecture Guide](./ARCHITECTURE.md) — Core package structure and data flow
- [Algorithm Library](./algorithms/) — Deep dives on supported algorithms
- [Policy Reference](./policies.md) — Algorithm comparison and complexity notes
- [Evaluation Methods](./evaluation.md) — Offline policy evaluation with IPS, DR, and NCIS
- [Advanced Features](./advanced_features.md) — Top-k decisions, abstention, constraints, normalization, and drift
- [Contributing Guide](../CONTRIBUTING.md) — Development setup and standards

## Quick Example

```python
import numpy as np
from coba import BanditConfig, ClusterBandit

bandit = ClusterBandit(
    arms=["a", "b", "c"],
    config=BanditConfig(policy="lin_ucb", n_clusters=2),
)

contexts = np.random.default_rng(0).normal(size=(50, 3))
arms = ["a", "b", "c", "a", "b"] * 10
rewards = np.random.default_rng(1).random(50)

bandit.fit_offline(contexts, arms, rewards)
decision = bandit.decide(np.array([0.1, 0.2, 0.3]))
```

## Core Concepts

- `coba.bandit.ClusterBandit` — Main entry point
- `coba.policies.*` — Learning algorithms
- `coba.router.ClusterRouter` — KMeans context clustering and model routing
- `coba.evaluation.*` — Offline policy evaluation helpers
- `coba.continuous.*` — Continuous-action bandits

## Project Stats

| Metric | Value |
|--------|-------|
| Algorithms | 17 |
| Core package | `src/coba` |
| Tests | `tests/` |

## License

MIT — See [LICENSE](../LICENSE)
