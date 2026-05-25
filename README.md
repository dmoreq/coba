# COBA — Contextual Bandit Algorithms

[![CI Pipeline](https://github.com/yourusername/coba/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/coba/actions/workflows/ci.yml)

COBA is a Python library for experimenting with contextual bandit algorithms, offline evaluation, drift detection, and continuous-action policies.

## Quick Start

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --frozen

# Run tests
uv run pytest tests/ -p no:asyncio --ignore=tests/test_shared_sim.py
```

## Example

```python
import numpy as np
from coba import BanditConfig, ClusterBandit

bandit = ClusterBandit(
    arms=["email", "sms", "push"],
    config=BanditConfig(policy="lin_ucb", n_clusters=2),
)

contexts = np.random.default_rng(0).normal(size=(100, 4))
arms = ["email", "sms", "push"] * 34
rewards = np.random.default_rng(1).random(102)[:100]

bandit.fit_offline(contexts, arms[:100], rewards)
decision = bandit.decide(np.array([0.2, -0.1, 0.4, 0.8]))
print(decision.chosen_arm)
```

## Features

- Discrete contextual bandits with clustering-aware routing
- LinUCB, LinTS, Thompson Sampling, UCB1, Softmax, logistic, GP-UCB, tree ensembles, and sklearn-backed policies
- Offline policy evaluation with rejection sampling, IPS/NCIS, and doubly robust estimates
- Reward normalization, drift detection, constrained arm pull rates, and top-k decisions
- Continuous-action bandits with CATS-style action trees

## Project Structure

```text
coba/
├── src/coba/              # Core bandit library
│   ├── bandit.py          # ClusterBandit public API
│   ├── policies/          # Algorithm implementations
│   ├── continuous/        # Continuous-action bandits
│   ├── evaluation.py      # Offline evaluation helpers
│   └── drift.py           # Drift detection
├── tests/                 # Core library tests
└── docs/                  # Library documentation
```

## Development

```bash
make lint
make check-types
make test
make coverage
```

## Documentation

- [Architecture Guide](./docs/ARCHITECTURE.md)
- [Algorithm Reference](./docs/algorithms/)
- [Policy Reference](./docs/policies.md)
- [Evaluation Methods](./docs/evaluation.md)
- [Advanced Features](./docs/advanced_features.md)
- [Contributing Guide](./CONTRIBUTING.md)

## License

MIT
