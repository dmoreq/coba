# Softmax (Temperature-Scaled Exploration)

## Overview

**Type:** Stochastic / Exploration-Exploitation via Temperature Scaling
**Lesson:** Lesson 11 (Softmax Playlist Generation)
**Policy Type:** `softmax`
**Best for:** Scenarios requiring smooth stochastic exploration with tunable intensity

---

## How It Works

Softmax converts arm scores into a probability distribution via the softmax function, then samples an arm.

$$P(\text{arm} = a) = \frac{\exp(\text{score}_a / \tau)}{\sum_{a'} \exp(\text{score}_{a'} / \tau)}$$

The temperature parameter `τ` controls exploration intensity:
- **`τ` → 0** (cold): Deterministic (always pick best)
- **`τ` large** (hot): Uniform distribution (pure exploration)
- **`τ` = 1** (moderate): Balanced exploration-exploitation

---

## Key Hyperparameters

**`tau`** (Temperature)
- **Default:** 1.0
- **Range:** > 0
- **Effect:** Controls exploration sharpness
  - Small (0.1): Exploit more, explore rarely
  - Medium (1.0): Balanced
  - Large (10.0): Explore frequently, similar to uniform

---

## Example Usage

```python
from coba.policies.softmax import SoftmaxArmModel

model = SoftmaxArmModel(
    arm="option_A",
    rng=np.random.default_rng(42),
    tau=1.0  # temperature
)

# Before updates, all arms have score 0 → uniform distribution
score = model.score(context=None)  # Returns one sample from distribution

# After observing rewards, scores diverge
model.update(context=None, reward=1.0)  # High reward increases score
score_after = model.score(context=None)  # Sampling now favors this arm more
```

---

## When to Use

| Scenario | Recommendation |
|----------|---|
| **Smooth exploration** | ✅ Good choice |
| **Binary/sparse rewards** | ✅ Works well |
| **Need reproducible randomness** | ✅ Deterministic via RNG |
| **Want interpretable probabilities** | ✅ Softmax is intuitive |
| **Offline evaluation** | ⚠️ Requires logging propensities |

---

## Comparison with Other Context-Free Policies

| Policy | Mechanism | Exploration |
|--------|-----------|---|
| **UCB1** | Confidence bounds | Optimistic bonus |
| **Thompson** | Beta posterior sampling | Bayesian uncertainty |
| **Softmax** | Temperature scaling | Smooth stochastic |
| **Epsilon-Greedy** | Epsilon probability | Hard switching |

Softmax is softer than epsilon-greedy (smooth probability vs sharp cutoff) and simpler to tune than Thompson (just one parameter).

---

## Lesson Context

**Lesson 11: Softmax Playlist Generation**

Users control `tau` to see how temperature affects playlist generation:
- Low temp: Algorithm tends to recommend the same favorite songs
- High temp: Algorithm recommends diverse, unexpected songs
- Perfect for teaching the explore-exploit trade-off in a relatable domain (music)

---

## References

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed), §2.2 Softmax Action Selection
- Kuleshov & Precup, *Algorithms for Multi-Armed Bandit Problems* (2014)
