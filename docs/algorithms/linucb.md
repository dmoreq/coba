# LinUCB (Linear Upper Confidence Bound)

## 1. Overview

* **Type**: Deterministic / Optimism in the Face of Uncertainty
* **Class**: `LinUCBArmModel` (`policies/linucb.py`)
* **Best for**: Scenarios requiring predictable, auditable exploration behavior.

## 2. How It Works

LinUCB fits an online Ridge Regression per arm. The UCB score is composed of two parts: an exploitation term and an exploration bonus.

$$\text{score}(x) = x^\top \hat{\beta} + \alpha \sqrt{x^\top A^{-1} x}$$

where:
* $x^\top \hat{\beta}$ is the exploitation term (the expected reward).
* $\alpha \sqrt{x^\top A^{-1} x}$ is the exploration bonus, which grows larger for context vectors $x$ that the model hasn't seen often.

## 3. Key Hyperparameters

* `alpha`: Exploration multiplier (default 1.0). Higher → more exploration.
* `l2_lambda`: L2 regularization (default 1.0). Higher → more shrinkage.
* `gamma`: Discount factor for non-stationarity (default 1.0). Set `< 1.0` (e.g., 0.99) to forget old data and adapt to new trends.
