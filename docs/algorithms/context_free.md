# Context-Free Policies

## 1. Overview

These policies ignore the context vector $\mathbf{x}$ entirely and rely solely on aggregate arm statistics. They are a much smarter alternative to traditional A/B testing: instead of splitting traffic 50/50 until a fixed deadline, they continuously shift traffic toward the best-performing arm as evidence accumulates.

**Tip:** Pair with `n_clusters=1` in `ClusterBandit` when you have no useful context features.

## 2. UCB1 (Upper Confidence Bound)

* **Type**: Deterministic
* **Class**: `UCB1ArmModel` (`policies/ucb1.py`)
* **How it works**: Tracks the mean reward per arm, then adds an exploration bonus that shrinks as an arm is pulled more often. Arms that haven't been tried much receive a large bonus, forcing the algorithm to explore them.

$$\text{score}_i = \mu_i + \alpha \sqrt{\frac{2 \ln N}{n_i}}$$

where $\mu_i$ is the mean reward for arm $i$, $N$ is total pulls across all arms, $n_i$ is pulls for arm $i$, and `alpha` scales exploration width.

## 3. Thompson Sampling (Beta-Bernoulli)

* **Type**: Stochastic
* **Class**: `ThompsonArmModel` (`policies/thompson.py`)
* **How it works**: Maintains a $\text{Beta}(\alpha, \beta)$ posterior per arm — $\alpha$ counts successes, $\beta$ counts failures. At each step it samples one value from each arm's Beta distribution and picks the arm with the highest sample. Arms with fewer observations have wider, more uncertain distributions, so they get chosen more often to reduce uncertainty.
* **Constraints**: Rewards must be in $[0, 1]$ (e.g., conversion rate, binary accept/reject).

## 4. Example

```python
import numpy as np
from coba.policies.thompson import ThompsonArmModel

model = ThompsonArmModel(
    arm="variant_B",
    rng=np.random.default_rng(42),
    alpha_prior=1.0,
    beta_prior=1.0,
)

# Score does not require a context vector
score = model.score(context=None)
print(f"Sampled conversion rate: {score:.4f}")

# Observe a success (reward = 1.0) and update
model.update(context=None, reward=1.0)
# After enough successes, the Beta distribution shifts right (higher scores)
```
