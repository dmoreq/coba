# COBA — Contextual Bandit Algorithms

[![CI Pipeline](https://github.com/dmoreq/coba/actions/workflows/ci.yml/badge.svg)](https://github.com/dmoreq/coba/actions/workflows/ci.yml)

COBA is a Python library for experimenting with contextual bandit algorithms, offline evaluation, drift detection, and continuous-action policies.

## Install

```bash
# From PyPI (once published)
pip install coba

# Directly from GitHub
pip install git+https://github.com/dmoreq/coba.git

# Or with uv
uv add git+https://github.com/dmoreq/coba.git
```

## Quick Start

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

- 17 bandit policies — LinUCB, LinTS, Thompson Sampling, UCB1, Softmax, logistic, GP-UCB, tree ensembles, neural linear, and more
- Cluster routing — KMeans context partitioning with per-cluster arm models
- Offline policy evaluation — rejection sampling, IPS/NCIS, doubly robust estimates
- Continuous actions — CATS-style action trees for real-valued action spaces
- Reward normalization, drift detection, constrained pull rates, top-k decisions, abstention
- Model persistence with `save_bandit` / `load_bandit`

## Development

```bash
uv sync --frozen
make lint
make check-types
make test
```

## Documentation

- [Architecture Guide](./docs/ARCHITECTURE.md)
- [Policy Reference](./docs/policies.md)
- [Evaluation Methods](./docs/evaluation.md)
- [Advanced Features](./docs/advanced_features.md)
- [Contributing Guide](./CONTRIBUTING.md)

## License

MIT
