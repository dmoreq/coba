# Bug Report: `ClusterBandit` fails with `PolicyType.EPSILON_GREEDY`

**Library:** `coba` (https://github.com/dmoreq/coba)
**Commit:** `dd67767ba17347c9827d2ff9ece882eb9ec1bd66`
**File:** `coba/policies/sklearn_models.py`, line 53
**Related:** `coba/bandit.py`, line ~267 (`ClusterBandit.__init__`)
**Severity:** High — epsilon-greedy policy is unusable via `ClusterBandit`

---

## Description

Creating a `ClusterBandit` with `policy=PolicyType.EPSILON_GREEDY` raises a `ValueError` immediately:

```python
from coba import ClusterBandit, PolicyType

bandit = ClusterBandit(
    arms=["a", "b"],
    n_features=2,
    policy=PolicyType.EPSILON_GREEDY,
    epsilon=0.1,
)
```

```
ValueError: base_estimator must be provided for EpsilonGreedyArmModel
```

## Root Cause

In `coba/router.py`, `_build_epsgreedy` calls `EpsilonGreedyArmModel(arm, rng, base_estimator, epsilon)`, but `base_estimator` is `None` because `ClusterBandit.__init__` never passes a `base_estimator` parameter through to the `ClusterRouter`.

In contrast, the UCB1 path (`_build_ucb1`) passes through correctly because `UCB1ArmModel` doesn't require a `base_estimator` — it only needs `arm` and `rng`.

## Impact

Users of `ClusterBandit` cannot use epsilon-greedy. This affects the high-level API that is the library's primary entry point. The lower-level `BaseArmModel` subclasses work fine when instantiated directly.

## Suggested Fix

### Option A: Pass `base_estimator` through `ClusterBandit`

Add a `base_estimator` parameter to `ClusterBandit.__init__` and forward it to `ClusterRouter`:

```python
# bandit.py
class ClusterBandit:
    def __init__(self, ..., base_estimator=None, ...):
        ...
        self._router = ClusterRouter(
            ...,
            base_estimator=base_estimator,
            ...
        )
```

### Option B: Default to `RidgeRegression` in `_build_epsgreedy`

In `router.py`, when `base_estimator` is `None`, default to `RidgeRegression(...)`:

```python
# router.py
def _build_epsgreedy(arm, cfg, n_features, rng):
    base = cfg.base_estimator or RidgeRegression(...)
    return EpsilonGreedyArmModel(arm, rng, base, cfg.epsilon)
```

This is the friendlier fix since it makes the default "just work" without the user needing to know about the internal estimator.

## Full Traceback

```
File "coba/bandit.py", line 267, in __init__
    self._router = ClusterRouter(...)
File "coba/router.py", line 375, in __init__
    _build_arm_models(...)
File "coba/router.py", line 281, in _build_arm_models
    arm: _build_model_for_arm(arm, cfg, n_features, rng, ...)
File "coba/router.py", line 268, in _build_model_for_arm
    return builder(arm, cfg, n_features, rng)
File "coba/router.py", line 88, in _build_epsgreedy
    return EpsilonGreedyArmModel(...)
File "coba/policies/sklearn_models.py", line 53, in __init__
    raise ValueError("base_estimator must be provided for EpsilonGreedyArmModel")
```

## Workaround

Use the lower-level arm model directly instead of `ClusterBandit`:

```python
from coba.policies import EpsilonGreedyArmModel
from coba.policies.ridge import RidgeRegression
import numpy as np

model = EpsilonGreedyArmModel("a", np.random.default_rng(), RidgeRegression(l2_lambda=1.0), epsilon=0.1)
```

Or use a different policy type (`UCB1`, `THOMPSON`, `LIN_UCB`) that work correctly with `ClusterBandit`.
