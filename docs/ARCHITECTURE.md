# COBA Core Architecture

COBA is a domain-agnostic contextual bandit library. The core package operates on numpy arrays and plain Python values so it can be embedded in services, notebooks, simulations, or offline evaluation pipelines.

## Package Layout

```text
src/coba/
├── bandit.py              # ClusterBandit public API
├── router.py              # Cluster-aware routing and per-arm model orchestration
├── config.py              # BanditConfig
├── schemas.py             # Decision and stats records
├── policies/              # Algorithm implementations
├── continuous/            # Continuous-action bandits and CATS policy
├── evaluation.py          # Rejection sampling, DR, NCIS helpers
├── offpolicy.py           # IPS and doubly robust offline updates
├── drift.py               # Page-Hinkley drift detector
├── normalizer.py          # Reward normalization
└── persistence.py         # Save/load helpers
```

## Main Data Flow

```text
caller context
  → ClusterBandit.decide()
  → ClusterRouter.predict()/score_all()
  → per-arm policy model score()
  → BanditDecision
  → caller observes reward
  → ClusterBandit.update()
  → per-arm policy model update()
```

## Policy Model Boundary

All discrete arm models implement the `BaseArmModel` shape:

- `score(context, total_pulls)` ranks an arm for selection.
- `update(context, reward, weight)` incorporates feedback.
- `update_batch(contexts, rewards, weights)` supports offline fitting.
- `reset()` clears learned state.
- `clone()` creates independent arm model instances.

Context-free policies ignore the context vector. Contextual policies use linear, logistic, Gaussian-process, tree-ensemble, neural-linear, or sklearn-backed estimators.

## Offline Evaluation

`evaluation.py` and `offpolicy.py` provide rejection sampling, inverse propensity scoring, normalized clipped IPS, and doubly robust estimators for logged bandit data.

## Continuous Actions

`src/coba/continuous/` models continuous action spaces by partitioning the action interval into leaves and learning over those leaves with CATS-style local updates.

## Persistence

Use `save_bandit()` and `load_bandit()` for joblib-backed model snapshots.
