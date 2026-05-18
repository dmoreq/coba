# CATS (Continuous Action Tree Search)

## Overview

**Type:** Tree-based Continuous Action Search
**Lesson:** Lesson 15 (CATS Real-Time Bidding)
**Policy Type:** `cats`
**Best for:** Continuous action spaces (e.g., auction bids, prices, parameters)

---

## How It Works

Instead of discrete arms, CATS partitions the continuous action space `[a_min, a_max]` into a binary tree.

**Leaf Structure:**
- Each leaf is a contiguous range: `[lo, hi]`
- Thompson Sampling maintains a Beta posterior per leaf
- At decision time, sample from each leaf's posterior, pick the leaf with highest sample
- Return the leaf's midpoint as the continuous action

**Example Tree (depth=3, range=[0, 5]):**
```
                    [0, 5]
                   /      \
              [0, 2.5]    [2.5, 5]
              /    \        /    \
        [0,1.25] [1.25,2.5] [2.5,3.75] [3.75,5]
        ...
```

After observing a reward for action 1.2, the algorithm updates the Beta posterior for the leaf containing 1.2.

---

## Key Hyperparameters

**`a_min` / `a_max`**
- **Action range bounds**
- **Example:** `a_min=0.01` (minimum bid), `a_max=10.0` (maximum bid)

**`cats_depth`**
- **Tree depth (height)**
- **Default:** 6
- **Range:** 1–12
- **Effect:**
  - Depth 1: Only 2 leaves (top/bottom half)
  - Depth 6: 64 leaves (fine-grained partitioning)
  - Higher depth = finer resolution, but slower learning per leaf

---

## Example Usage

```python
from coba.continuous.bandit import ContinuousBandit
from coba.config import BanditConfig

# Real-time bidding: optimize bid price in range [0.01, 10.0]
config = BanditConfig(
    policy=PolicyType.CATS,
    cats_a_min=0.01,
    cats_a_max=10.0,
    cats_depth=6  # 64 leaves
)

bandit = ContinuousBandit(
    a_min=0.01,
    a_max=10.0,
    n_features=4,  # User, context features
    config=config
)

# Decide on bid
context = np.array([user_value, time_of_day, competition, pacing])
decision = bandit.decide(context)
chosen_bid = decision.chosen_action  # Float in [0.01, 10.0]

# Observe outcome (win rate, revenue, etc.)
bandit.update(context=context, arm=chosen_bid, reward=0.95)
```

---

## Continuous vs Discrete Bandits

| Aspect | Discrete (LinUCB) | Continuous (CATS) |
|--------|---|---|
| **Actions** | `["A", "B", "C"]` | `0.0–10.0` (infinite) |
| **Learning** | Per-arm linear model | Per-leaf Thompson |
| **Decision time** | O(n_arms × d) | O(tree_depth × d) |
| **Use case** | Content selection | Price/bid optimization |

---

## Lesson Context

**Lesson 15: CATS Real-Time Bidding**

Users optimize ad bid prices in a simulated RTB environment:
- Context: User value, time-of-day, competition, pacing
- Action: Bid price (continuous, $0–$10)
- Reward: Probability of winning the impression
- Goal: Find the sweet spot (high bid → always win → waste money; low bid → never win)

**Interactive controls:**
- Adjust `cats_depth` to see finer vs coarser partitioning
- Observe which bid ranges the tree explores
- Visualize leaf scores in real-time (shown in `leafScores` endpoint)

---

## Algorithm Details

### Tree Construction
The tree is a **complete binary tree** with `2^depth` leaves.

Node `i` at depth `d` covers range:
$$[\text{a\_min} + i \cdot \text{width}, \text{a\_min} + (i+1) \cdot \text{width}]$$

where width = `(a_max - a_min) / 2^depth`

### Leaf Selection
At each decision step:
1. For each leaf, sample from its Beta posterior
2. Pick the leaf with the highest sample value
3. Return the midpoint as the continuous action

### Learning
After observing reward `r` for action `a`:
1. Find the leaf containing `a`
2. Update the Beta posterior: `alpha += reward`, `beta += (1 - reward)`
3. Other leaves' posteriors unchanged

---

## When to Use

| Scenario | Recommendation |
|----------|---|
| **Discrete arms** | ❌ Use LinUCB instead |
| **Continuous optimization** | ✅ Perfect fit |
| **Real-time pricing** | ✅ Ideal |
| **A/B testing with values** | ✅ Good choice |
| **Parameter tuning** | ✅ Suitable |
| **Thousands of actions** | ✅ Efficient vs alternatives |

---

## Comparison with Alternatives

| Approach | Method | Complexity |
|----------|--------|---|
| **Grid search + LinUCB** | Discretize into K buckets, use K-arm LinUCB | O(d² × K) |
| **Gaussian Process** | GP posterior over continuous function | O(n²) |
| **Gradient-based** | Bandit with gradient estimates (tricky) | O(d) per iteration |
| **CATS** | Binary tree + Thompson per leaf | O(depth × d) |

CATS is **simpler than GP** (Thompson vs EM), **faster than grid search** (logarithmic depth), and **avoids gradient instability**.

---

## Leaf Scores Endpoint

The backend `/api/sessions/{id}/leaf-scores` returns:

```json
{
  "leaves": [
    {
      "index": 0,
      "lo": 0.01,
      "hi": 5.0,
      "midpoint": 2.505,
      "ucb": 1.234  // Upper confidence bound for this leaf
    },
    // ...
  ],
  "active_leaf": 5,
  "sampled_action": 3.256
}
```

**Visualization:** The lesson shows which leaf was chosen and its score.

---

## Lesson Implementation Notes

The frontend component renders:
- **Leaf UCB scores** as bars (higher = more promising)
- **Active leaf highlight** (sampled in this step)
- **Action range** with chosen point marked

---

## References

- Kannan et al., *Bandits with Delayed, Aggregated Anonymous Feedback* (ICML 2018)
- See also: [Advanced Features](./advanced_features.md) for production constraints (min_pull_rates, abstention)
