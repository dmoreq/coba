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

## Advanced Patterns

### Multi-Objective via Reward Scalarization

To optimize multiple metrics simultaneously, compute a composite reward before calling `update()`:

```python
w_primary   = 0.7
w_secondary = 0.3

normalized_primary   = raw_primary / max_primary
normalized_secondary = raw_secondary / max_secondary

composite_reward = (w_primary * normalized_primary) + (w_secondary * normalized_secondary)
bandit.update(context=ctx, arm=chosen_arm, reward=composite_reward)
```

This keeps all learning engines at full speed while multi-objective logic lives entirely in the application layer.

### Domain Integration

Build a **Domain Facade** that translates your domain objects into the raw `numpy` arrays COBA expects. COBA never needs to know what the context features or arm identifiers mean — that mapping lives entirely in your application layer.

> For a runnable Quick Start, see the [README](../README.md).
