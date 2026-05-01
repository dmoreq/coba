# Advanced Features Guide

This guide covers six features added to `ClusterBandit` beyond the core decide/update loop: ranked arm selection, confidence-based abstention, minimum pull-rate floors, per-arm gamma overrides, running reward normalization, and reward drift detection.

---

## 1. `decide_top_k` — Ranked Arm Selection

**API**: `bandit.decide_top_k(context, k) -> list[tuple[Arm, float]]`

Returns the top-`k` arms ranked by score for the given context, as a list of `(arm, score)` tuples in descending order. `k` is silently clamped to `len(arms)` if it exceeds the number of available arms.

```python
import numpy as np

top3 = bandit.decide_top_k(context, k=3)
# [(arm_name, score), (arm_name, score), (arm_name, score)]

primary, fallback_1, fallback_2 = [arm for arm, _ in top3]
```

**Cold start**: before `fit()`, returns the first `k` arms with score `0.0`.

**Use cases**:

| Use case | Example |
|----------|---------|
| Ranked recommendation slate | Show top-3 articles instead of 1 |
| Fallback chain | Try best arm first; if unavailable, try second |
| Diversity exploration | Sample from top-k instead of always taking the top-1 |

---

## 2. Confidence-Based Abstention

**API**: `bandit.decide(context, min_confidence_gap=0.0)`

When the score gap between the best and second-best arm is smaller than `min_confidence_gap`, the bandit **abstains** — it returns a `BanditDecision` with `chosen_arm=None` and `abstained=True`. The caller is responsible for applying a fallback.

Setting `min_confidence_gap=0.0` (the default) disables abstention completely.

```python
decision = bandit.decide(context, min_confidence_gap=0.1)

if decision.abstained:
    # Bandit is uncertain — delegate to a rule-based policy
    arm = rule_based_fallback(context)
else:
    arm = decision.chosen_arm
```

**Always check `decision.abstained`** before using `decision.chosen_arm` when a non-zero gap threshold is set — `chosen_arm` is `None` on abstention.

**Tuning `min_confidence_gap`**:

| Value | Effect |
|-------|--------|
| `0.0` | Never abstains (default) |
| Small (e.g. `0.05`) | Abstains only when arms are nearly tied |
| Large (e.g. `0.5`) | Abstains frequently — requires robust fallback |

---

## 3. `min_pull_rates` — Guaranteed Arm Exploration

**Constructor parameter**: `ClusterBandit(min_pull_rates={"arm_name": fraction})`

Maps each arm to a minimum required fraction of total decisions. When an arm's actual pull rate falls below its floor, the bandit restricts the candidate set to only the under-pulled arms and picks the best among them.

```python
bandit = ClusterBandit(
    arms=["control", "variant_A", "new_feature"],
    n_features=5,
    min_pull_rates={
        "new_feature": 0.05,   # guarantee ≥ 5 % of traffic
        "variant_A":   0.10,   # guarantee ≥ 10 % of traffic
    },
)
```

**Validation rules** (enforced at construction time):
- Each rate must be in `(0.0, 1.0]`
- Sum of all rates must be `≤ 1.0`
- All arm names must exist in `arms`

**How it works**: after `fit()`, every `decide()` call checks `arm_pulls / total_decisions` per arm. If any arm is below its floor, only those arms compete for the decision — the greedy winner among the under-pulled arms is chosen.

**Use cases**: new arm launch with business traffic floor, regulatory minimum exposure, guaranteed A/B split.

---

## 4. Per-Arm Gamma Override in `add_arm()`

**API**: `bandit.add_arm(arm, warm_start_from=None, gamma=None)`

The bandit-level `gamma` applies a uniform discount to all arms' reward histories. When adding a new arm to a live system, you may want that arm to **adapt faster** to its own reward distribution without touching the existing arms.

```python
# Existing bandit trained with gamma=1.0 (stationary reward assumed)
bandit.add_arm(
    "new_variant",
    warm_start_from="control",  # copy control's model as warm start
    gamma=0.9,                  # new arm forgets old data 10× faster
)
```

| `gamma` | Effect |
|---------|--------|
| `None` | Inherits bandit-level `gamma` (default) |
| `1.0` | No discounting — full history retained |
| `0.9` | Each update down-weights prior observations by 10 % |
| `< 0.95` | Aggressive forgetting — useful for rapidly shifting distributions |

**Note**: `gamma` affects the Sherman-Morrison update in the arm's ridge model, not a rolling window. The effective influence of an observation `t` steps ago is proportional to `gamma^t`.

---

## 5. `RewardNormalizer` — Running Reward Scaling

**Import**: `from coba.normalizer import RewardNormalizer`

Bandit arm models (especially Thompson Sampling and Logistic) assume rewards in `[0, 1]`. Real business metrics (revenue, dwell time, click depth) are rarely in that range. `RewardNormalizer` tracks running statistics and scales each reward before it reaches the bandit.

```python
from coba.normalizer import RewardNormalizer

normalizer = RewardNormalizer(mode="minmax")  # or "zscore"

for raw_reward in revenue_stream:
    normed = normalizer.update_and_normalize(raw_reward)
    bandit.update(context=ctx, arm=arm, reward=normed)
```

### Modes

| Mode | Output range | When to use |
|------|-------------|-------------|
| `"minmax"` | `[0, 1]` (clipped) | CTR, conversion rate, bounded metrics |
| `"zscore"` | ~`(-3, 3)` | Revenue, engagement time, linear contextual bandits |

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `"minmax"` | Normalization mode |
| `decay` | `0.999` | EMA decay for running stats — higher = more history retained |
| `clip` | `True` | (minmax only) Clamp output to `[0, 1]` for unseen extremes |

### Two Methods

```python
# Updates running stats AND returns normalized value — use during training
normed = normalizer.update_and_normalize(raw_reward)

# Normalizes WITHOUT updating stats — use for evaluation data
normed = normalizer.normalize(raw_reward)
```

---

## 6. `PageHinkleyDetector` — Reward Drift Detection

**Import**: `from coba.drift import PageHinkleyDetector`

The Page-Hinkley test monitors a scalar stream and flags a distributional shift when the cumulative deviation from a reference mean exceeds a threshold. The two-sided variant detects both reward improvements and degradations.

### Automatic Integration

Enable via `ClusterBandit` constructor — no manual wiring needed:

```python
bandit = ClusterBandit(
    arms=["A", "B", "C"],
    n_features=5,
    enable_drift_detection=True,
    drift_delta=0.005,    # minimum detectable change magnitude
    drift_lambda=50.0,    # threshold before alarm fires
)
# Detected drift on arm X automatically resets X's cluster models
```

### Manual Usage

```python
from coba.drift import PageHinkleyDetector

detector = PageHinkleyDetector(delta=0.01, lambda_=30.0, alpha=0.999)

for reward in reward_stream:
    if detector.update(reward):
        bandit.reset_arm("arm_name")   # reset that arm's models across all clusters
        detector.reset()               # keep mean, restart cumulative sums
```

### Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `delta` | `0.005` | Minimum magnitude of change to detect. Lower → more sensitive (more false alarms) |
| `lambda_` | `50.0` | Detection threshold. Higher → fewer false alarms, slower detection |
| `alpha` | `0.999` | EMA weight for reference mean. Higher → more stable baseline |

### Reset vs Full Reset

| Method | What resets | When to use |
|--------|-------------|-------------|
| `reset()` | Cumulative sums only; keeps mean | After handling drift — continue with same reference |
| `full_reset()` | Everything including mean | Arm fully removed and re-added |
