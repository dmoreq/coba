# Advanced Features Guide

This guide covers six features added to `ClusterBandit` beyond the core decide/update loop: ranked arm selection, confidence-based abstention, minimum pull-rate floors, per-arm gamma overrides, running reward normalization, and reward drift detection.

> **Web Platform Note:** COBA Web implements many of these features as interactive lesson examples. See [Lesson 14 (Offline Evaluation)](../web/frontend/components/lesson/OfflineEvaluationLesson.tsx) for IPS/DR/NCIS, [Lesson 13 (Drift Detection)](../web/frontend/components/lesson/DriftDetectionLesson.tsx) for PageHinkley in action, and [Lesson 12 (Sliding-Window LinUCB)](../web/frontend/components/lesson/SlidingWindowLinUCBLesson.tsx) for adaptive exploration under drift.

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

---

## 7. Batch Operations — `decide_batch` and `update_batch`

**Import**: `from coba.bandit import ClusterBandit`

For high-throughput systems, COBA provides vectorized decision and update methods that process multiple contexts in one call, reducing function call overhead and enabling better CPU parallelization.

### `decide_batch` — Vectorized Decision Making

**API**: `bandit.decide_batch(contexts: np.ndarray) -> list[BanditDecision]`

Makes decisions for multiple contexts in one call.

```python
import numpy as np

# Batch of 1000 contexts
contexts = np.random.randn(1000, n_features)

# Single vectorized call
decisions = bandit.decide_batch(contexts)

# Each decision is a BanditDecision object
for decision in decisions:
    chosen_arm = decision.chosen_arm
    score = decision.score
    if decision.abstained:
        # handle fallback
```

**Benefits:**
- Single KMeans cluster assignment pass for all contexts
- Bulk arm score computation
- Faster than 1000 sequential `decide()` calls

**Returns**: `list[BanditDecision]` where each element corresponds to one context in input order.

### `update_batch` — Vectorized Learning

**API**: `bandit.update_batch(contexts: np.ndarray, arms: list[Arm], rewards: np.ndarray, propensities: np.ndarray | None = None)`

Updates the bandit model on multiple observations in one call.

```python
# 1000 decisions made
contexts = np.random.randn(1000, n_features)
chosen_arms = [decision.chosen_arm for decision in decisions]
rewards = np.random.rand(1000)  # observed outcomes
propensities = np.full(1000, 1.0 / n_arms)  # logging policy prob

# Single batch update
bandit.update_batch(
    contexts=contexts,
    arms=chosen_arms,
    rewards=rewards,
    propensities=propensities,  # optional, default 1.0 per observation
)
```

**Benefits:**
- Single pass through all clusters
- Ridge matrix updates batched per cluster
- Faster than 1000 sequential `update()` calls
- Supports off-policy correction via `propensities`

**Off-policy note**: If using logged data where the logging policy was biased, pass `propensities` to correct via inverse propensity scoring (IPS).

### `fit_offline` — Bootstrap from Historical Data

**API**: `bandit.fit_offline(contexts: np.ndarray, decisions: np.ndarray, rewards: np.ndarray, propensities: np.ndarray | None = None)`

Bootstrap the bandit from historical logs before going live. Automatically applies IPS correction if propensities are provided.

```python
import numpy as np

# Load historical data
n_samples = 100_000
contexts = load_features("historical_data.parquet", n_samples)
decisions = load_decisions("historical_data.parquet", n_samples)
rewards = load_rewards("historical_data.parquet", n_samples)
propensities = load_propensities("historical_data.parquet", n_samples)

# Fit on historical data (with IPS bias correction)
bandit = ClusterBandit(
    arms=["A", "B", "C"],
    n_features=10,
    policy="linucb",
    n_clusters=5,
)

bandit.fit_offline(
    contexts=contexts,
    decisions=decisions,
    rewards=rewards,
    propensities=propensities,  # IPS correction applied
)

# Now ready to serve live
ctx = np.array([...])
decision = bandit.decide(ctx)
```

**What it does:**
1. Fits KMeans clusters on context data
2. For each context, assigns to its cluster
3. Updates per-arm models in each cluster with observed (context, arm, reward)
4. If propensities provided, applies inverse propensity weighting to correct logging bias

**Use cases:**
- Starting a bandit from a rule-based system's historical decisions
- A/B test data → contextual bandit initialization
- Batch learning from logs before real-time serving

---

## 8. Model Persistence — `save_bandit` and `load_bandit`

**Import**: `from coba.persistence import save_bandit, load_bandit`

COBA bandits can be serialized to disk for offline analysis, model versioning, and production deployment.

### Saving a Bandit

**API**: `save_bandit(bandit: ClusterBandit, path: str | Path) -> None`

Persists a trained `ClusterBandit` to disk using joblib compression.

```python
from coba.persistence import save_bandit
import numpy as np

# Train a bandit
bandit = ClusterBandit(
    arms=["A", "B", "C"],
    n_features=5,
    policy="linucb",
)

for context, arm, reward in training_data:
    bandit.update(context, arm, reward)

# Save to disk
save_bandit(bandit, "/models/bandit_v1.0.joblib")
# Creates parent directories if needed
```

**Details:**
- Uses joblib with compress level 3 (good compression, reasonable speed)
- Automatically creates parent directories
- Overwrites existing file
- File size: typically 10–100 MB for a trained bandit (KMeans clusters + ridge matrices)

### Loading a Bandit

**API**: `load_bandit(path: str | Path) -> ClusterBandit`

Deserializes a saved bandit from disk.

```python
from coba.persistence import load_bandit

# Load a trained model
bandit = load_bandit("/models/bandit_v1.0.joblib")

# Use immediately for serving
ctx = np.array([...])
decision = bandit.decide(ctx)
```

**Error handling:**
- Raises `FileNotFoundError` if path doesn't exist
- Raises joblib exception if file is corrupted or incompatible

### Use Cases

| Scenario | Example |
|----------|---------|
| Model versioning | Train multiple bandits, save each with version tag |
| Offline analysis | Save trained model, analyze clusters + arm weights in notebook |
| Production deployment | Train offline, save, deploy saved model to production servers |
| A/B testing policies | Save two trained bandits, serve from each to different user cohorts |
| Disaster recovery | Regular snapshots for rollback if new model performs poorly |

### Backward Compatibility

Legacy code may use `save_model` and `load_model` — these are aliases:

```python
from coba.persistence import save_model, load_model  # equivalent to save_bandit, load_bandit
```

---

## 9. `BanditDecision` — Understanding the Decision Output

**Import**: `from coba.schemas import BanditDecision`

Every `bandit.decide()` call returns a `BanditDecision` object containing not just the chosen arm but also detailed decision reasoning.

### Full Field Reference

```python
decision = bandit.decide(context)
```

| Field | Type | Example | Notes |
|-------|------|---------|-------|
| `chosen_arm` | `str \| None` | `"variant_B"` | `None` if `abstained=True` |
| `score` | `float` | `0.85` | The final score for the chosen arm |
| `abstained` | `bool` | `False` | `True` if confidence gap too small (see min_confidence_gap) |
| `all_scores` | `dict[str, float]` | `{"A": 0.7, "B": 0.85, "C": 0.6}` | Scores for all arms |
| `mean_estimate` | `float \| None` | `0.75` | Exploitation component (expected reward); LinUCB/LinTS only |
| `confidence_width` | `float \| None` | `0.10` | Exploration component (uncertainty bonus); LinUCB/LinTS only |

### Using Decision Components

#### Simple case — use the chosen arm:
```python
decision = bandit.decide(ctx)
arm = decision.chosen_arm
if not decision.abstained:
    serve(arm)
else:
    serve_fallback()
```

#### Advanced case — exploit vs explore decomposition (LinUCB/LinTS):
```python
decision = bandit.decide(ctx)

if decision.mean_estimate is not None and decision.confidence_width is not None:
    exploitation_ratio = decision.mean_estimate / decision.score
    exploration_ratio = decision.confidence_width / decision.score

    logger.info(f"Chose {decision.chosen_arm} "
                f"(exploit: {exploitation_ratio:.1%}, explore: {exploration_ratio:.1%})")
```

#### Monitoring — check all arm scores:
```python
decision = bandit.decide(ctx)

for arm, score in decision.all_scores.items():
    logger.gauge(f"arm.{arm}.score", score)
```

### Policies and Field Availability

Not all policies populate `mean_estimate` and `confidence_width`:

| Policy | `mean_estimate` | `confidence_width` | Notes |
|--------|-----------------|-------------------|-------|
| LinUCB | ✅ | ✅ | Always available |
| LinTS | ✅ | ❌ | Thompson sample doesn't expose width |
| LinUCB-Hybrid | ✅ | ✅ | Same as LinUCB |
| LinUCB-SW | ✅ | ✅ | Same as LinUCB |
| GP-UCB | ❌ | ❌ | GP scores opaque |
| Neural Linear | ❌ | ❌ | MLP backbone outputs opaque |
| Logistic UCB | ❌ | ❌ | Logistic link obscures decomposition |
| Thompson, UCB1, Softmax | ❌ | ❌ | Context-free policies have no decomposition |

**Design note**: Even when fields are `None`, `chosen_arm` and `score` are always populated.
