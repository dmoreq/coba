# Logistic Bandits (Laplace Approximation)

## 1. Overview

* **Type**: Generalized Linear Model Bandit
* **Classes**: `LogisticUCBArmModel`, `LogisticTSArmModel` (`policies/logistic.py`)
* **Best for**: Binary reward environments — e.g., conversion rate, click-through rate, acceptance probability.

## 2. How It Works

Traditional linear bandits assume rewards range continuously. For binary outcomes (0 or 1), predicting outside this range is problematic. Logistic bandits apply a sigmoid function to map predictions to a strict $[0, 1]$ probability.

To keep computation fast in an online setting, COBA uses an **Online Laplace Approximation** (1-step Newton-Raphson + Sherman-Morrison) to maintain the inverse Hessian $H^{-1}$ in $O(d^2)$ time.

* **Logistic UCB**: Applies the confidence bound in logit space before passing it through the sigmoid.
* **Logistic TS**: Samples from the Gaussian posterior over the logistic weights before applying the sigmoid.

## 3. Key Hyperparameters

* `alpha` (UCB only): Exploration width in logit space (default 1.0). Higher → wider confidence bound.
* `v_sq` (TS only): Posterior variance multiplier (default 1.0). Higher → more stochastic exploration.
* `l2_lambda`: L2 regularization on the Hessian approximation (default 1.0).
* `gamma`: Discount factor for non-stationarity (default 1.0). Set `< 1.0` (e.g., 0.99) to down-weight old observations.

## 4. Variants

**Logistic Thompson Sampling (LogisticTS)**
- Bayesian variant: samples from Gaussian posterior over logistic coefficients
- Same computational complexity as LogisticUCB (Laplace approximation)
- Better for uncertainty quantification; tends to explore more aggressively
- Lesson: Not featured (available as policy), but you can experiment via Python API

## 5. Example Usage

```python
import numpy as np
from coba.policies.logistic import LogisticUCBArmModel

model = LogisticUCBArmModel(arm="variant_B", n_features=5, alpha=1.0)

# Binary reward: 1.0 = converted, 0.0 = not converted
model.update(np.array([1.0, 0.5, -0.2, 0.0, 0.1]), reward=1.0)

# Returns a probability in (0, 1)
prob = model.score(np.array([1.0, 0.5, -0.2, 0.0, 0.1]))
print(f"Estimated conversion probability: {prob:.4f}")
```
