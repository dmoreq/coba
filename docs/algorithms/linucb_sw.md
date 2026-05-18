# Sliding-Window LinUCB

## Overview

**Type:** Deterministic, Time-Adaptive
**Lesson:** Lesson 12 (Sliding-Window LinUCB Flash Sale)
**Policy Type:** `linucb_sw`
**Best for:** Non-stationary environments where recent data is more relevant

---

## How It Works

Sliding-Window LinUCB is LinUCB that only keeps the most recent `W` observations per arm. Older data is discarded.

**Standard LinUCB:**
$$\text{score}(x) = x^\top \hat{\beta} + \alpha \sqrt{x^\top A^{-1} x}$$

**Sliding-Window LinUCB:**
- Maintain ridge regression for each arm using only the last `W` observations
- When the window is full, drop the oldest observation and add the new one
- Same score function as LinUCB, but on windowed data

---

## Key Hyperparameters

**`window_size` (or `linucb_sw_window`)**
- **Default:** 200
- **Range:** > 0 (typically 100–1000)
- **Effect:** How many recent observations to retain
  - Small window (50): Adapt quickly to drift, but noisier
  - Large window (1000): Stable but slow to adapt
  - Rule of thumb: ~3-5× expected number of updates per arm

---

## Example Usage

```python
from coba.policies.linucb import SlidingWindowLinUCBArmModel

model = SlidingWindowLinUCBArmModel(
    arm="variant_A",
    n_features=5,
    window_size=200,  # Keep last 200 observations
    alpha=1.0,
    l2_lambda=1.0
)

# First 200 observations fill the window
for i in range(200):
    context = np.random.randn(5)
    model.update(context, reward=np.random.rand())

# Starting at observation 201, oldest data is dropped
model.update(context, reward=0.9)  # Replaces the oldest (200 steps ago)
```

---

## Drift Adaptation Example

**Use Case: Flash Sale**

Product demand shifts throughout a 24-hour flash sale:
- **6 AM:** Low demand, high discounts work (reward = discount * demand)
- **12 PM:** Peak demand, discounts less effective
- **6 PM:** Tail-off, aggressive discounts return

Standard LinUCB learns a global model that averages over time. Sliding-Window LinUCB only looks at recent patterns → adapts faster.

```
Time      | Demand | Best Discount | LinUCB Score | SW-LinUCB Score
----------|--------|---------------|--------------|----------------
06:00     | 10%    | 50%           | (outdated)   | 50% ← fresh data
12:00     | 100%   | 5%            | (outdated)   | 5%  ← recent peak
18:00     | 30%    | 40%           | (outdated)   | 40% ← adapting
```

---

## When to Use

| Scenario | Recommendation |
|----------|---|
| **Stationary rewards** | ❌ Not needed; use standard LinUCB |
| **Slowly drifting rewards** | ⚠️ Medium window size |
| **Rapid concept drift** | ✅ Essential; small window |
| **Seasonal patterns** | ✅ Good choice |
| **Real-time auctions, pricing** | ✅ Ideal |

---

## Comparison with Drift Detection

| Approach | Mechanism | When to reset |
|----------|-----------|---|
| **Sliding-Window** | Discard old data continuously | Always (every step) |
| **Drift Detection** | Monitor for shift, reset on alarm | Only on detected change |
| **Adaptive γ** | Discount old observations | Gamma < 1.0 |

- **Sliding-Window:** Simple, no detection tuning
- **Drift Detection:** Efficient (don't reset unless needed), but requires threshold tuning
- **Adaptive γ:** Smooth fade-out (not hard cutoff)

---

## Lesson Context

**Lesson 12: Sliding-Window LinUCB Flash Sale**

Users control the window size and observe how drift adaptation works:
- **Small window (50):** Tracks demand swings closely but noisier
- **Large window (500):** Smoother but lags behind drift
- **Optimal window (200):** Balance between responsiveness and stability

Compare to a standard LinUCB run to see the difference in real-time adaptation.

---

## Implementation Notes

- **Efficiency:** Removing old observations from ridge regression requires updating the inverse matrix (Sherman-Morrison formula backward step). O(d²) per removal.
- **Numerical stability:** Window size should be >> d (feature dimension) to avoid rank deficiency.
- **Memory:** O(window_size × d) instead of O(total_observations × d).

---

## References

- Jadbabaie et al., *Online Optimization Under Time-Varying Distributions* (2016)
- Besbes et al., *Stochastic Optimization under Time-Varying Distributions* (2015)
- See also: [Drift Detection](./advanced_features.md#6-pagehinkleydetector--reward-drift-detection)
