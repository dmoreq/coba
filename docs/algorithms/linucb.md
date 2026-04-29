# LinUCB (Linear Upper Confidence Bound)

## 1. Overview

* **Type**: Deterministic / Optimism in the Face of Uncertainty
* **Class**: `LinUCBArmModel` (`policies/linucb.py`)
* **Best for**: Scenarios requiring predictable, auditable exploration behavior.

## 2. How It Works

LinUCB fits an online Ridge Regression per arm. The score combines an exploitation term (expected reward) with an exploration bonus (uncertainty):

$$\text{score}(x) = \underbrace{x^\top \hat{\beta}}_{\text{exploit}} + \underbrace{\alpha \sqrt{x^\top A^{-1} x}}_{\text{explore}}$$

The exploration bonus $\alpha \sqrt{x^\top A^{-1} x}$ is large for context vectors the model hasn't seen much. As an arm accumulates observations at context $x$, $A$ grows and the bonus shrinks, shifting the balance toward exploitation.

## 3. Key Hyperparameters

* `alpha`: Exploration width (default 1.0). Higher → more exploration. Typical range: 0.5–2.0.
* `l2_lambda`: L2 regularization (default 1.0). Higher → more shrinkage toward zero.
* `gamma`: Discount factor for non-stationarity (default 1.0). Set `< 1.0` (e.g., 0.99) to down-weight old observations and adapt to distribution shifts.

## 4. Example

```python
import numpy as np
from coba.policies.linucb import LinUCBArmModel

model = LinUCBArmModel(arm="variant_A", n_features=3, alpha=1.0, gamma=0.99)

context = np.array([0.8, 0.2, 0.9])

# Before any updates the exploration bonus dominates — the model is curious.
score_before = model.score(context)
print(f"Score (cold): {score_before:.4f}")

model.update(context, reward=1.0)

# After an update the exploitation term contributes more; score is more precise.
score_after = model.score(context)
print(f"Score (after 1 update): {score_after:.4f}")
```
