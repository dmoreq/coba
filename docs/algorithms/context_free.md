# Context-Free Policies

## 1. Overview

These policies ignore the context vector $\mathbf{x}$ entirely and rely solely on aggregate arm statistics. They are essentially a much smarter version of traditional A/B testing.
**Tip:** Pair these with `n_clusters=1` in `ClusterBandit` for the best results.

## 2. UCB1 (Upper Confidence Bound)

* **Type**: Deterministic
* **Class**: `UCB1ArmModel` (`policies/ucb1.py`)
* **How it works**:

$$\text{score}_i = \mu_i + \alpha \sqrt{\frac{2 \ln N}{n_i}}$$

where $\mu_i$ is the empirical mean, $N$ is total pulls across all arms, and $n_i$ is the specific arm $i$'s pull count.

## 3. Thompson Sampling (Beta-Bernoulli)

* **Type**: Stochastic
* **Class**: `ThompsonArmModel` (`policies/thompson.py`)
* **How it works**: Maintains a $\text{Beta}(\alpha, \beta)$ posterior distribution per arm.
* **Constraints**: Rewards must be strictly bounded in $[0, 1]$.
* **Execution**: Samples from the Beta posterior at each call. Highly effective in practice for context-free scenarios.
