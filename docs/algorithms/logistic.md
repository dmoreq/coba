# Logistic Bandits (Laplace Approximation)

## 1. Overview

* **Type**: Generalized Linear Model Bandit
* **Classes**: `LogisticUCBArmModel`, `LogisticTSArmModel` (`policies/logistic.py`)
* **Best for**: Binary reward environments — e.g., conversion rate, click-through rate, trip booking probability.

## 2. How It Works

Traditional linear bandits assume the reward ranges continuously. For binary outcomes (0 or 1), predicting outside this range is problematic. Logistic bandits apply a sigmoid function to map predictions to a strict $[0, 1]$ probability.

To keep computation fast in an online setting, COBA uses an **Online Laplace Approximation** (1-step Newton-Raphson + Sherman-Morrison) to maintain the inverse Hessian $H^{-1}$ in $O(d^2)$ time. 

* **Logistic UCB**: Applies the confidence bound in logit space before passing it through the sigmoid.
* **Logistic TS**: Samples from the Gaussian posterior over the logistic weights before applying the sigmoid.

## 3. Example Usage

```python
from coba.policies.logistic import LogisticUCBArmModel
import numpy as np

model = LogisticUCBArmModel(arm="price_1_2x", n_features=5, alpha=1.0)

# Binary reward: 1.0 = trip booked, 0.0 = no booking
model.update(np.array([1.0, 0.5, -0.2, 0.0, 0.1]), reward=1.0)

# Returns a strict probability between 0 and 1
prob_ucb = model.score(np.array([1.0, 0.5, -0.2, 0.0, 0.1])) 
```
