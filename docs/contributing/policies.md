# Adding a Policy

Policies implement the `BanditPolicy` Protocol and optionally `DebugSnapshotProvider`. Here's how to add a new bandit algorithm.

## 1. Choose Your Policy Type

- **Context-free** (`BanditPolicy[Any, Any]`): ignores context, uses arm statistics only
- **Contextual** (`BanditPolicy[str, dict[str, Any]]`): uses feature vectors from context

## 2. Implement the Policy

Create `src/web/policies/my_policy.py`:

```python
from collections.abc import Sequence
from typing import Any
from web.contracts import BanditPolicy, DebugSnapshotProvider


class MyPolicy(BanditPolicy[str, dict[str, Any]], DebugSnapshotProvider):
    def __init__(self, param: float = 1.0, seed: int = 0) -> None:
        if param <= 0.0:
            raise ValueError("param must be > 0")
        self.param = param
        self._rng = random.Random(seed)
        # initialize your state

    def reset(self) -> None:
        # clear all state for a new run

    def select_arm(self, context: dict[str, Any], arms: Sequence[str]) -> str:
        # return the chosen arm_id

    def update(self, context: dict[str, Any], arm: str, reward: float) -> None:
        # update internal state based on observed reward

    def get_debug_snapshot(self) -> dict[str, Any]:
        return {
            "policy": "my_policy",
            "param": self.param,
            # expose internals for debugger
        }
```

**Requirements:**
- Parameter validation in `__init__` (raise `ValueError` for invalid values).
- Deterministic seeding: initialize `random.Random(seed)` or `numpy.random.RandomState(seed)`.
- `select_arm` must raise `ValueError` if `arms` is empty.
- Implement `DebugSnapshotProvider` — all policies must support debugger.

## 3. Register in Factory

In `src/web/policy_factory.py`:

1. Import your policy class from `web.policies`.
2. Add an `if` branch:
```python
if policy_id == "my_policy":
    return MyPolicy(
        param=float(params.get("param", 1.0)),
        seed=seed,
    )
```

## 4. Register Capabilities

In `src/web/policy_capabilities.py`:

```python
"my_policy": PolicyCapability(
    "my_policy", "context_free", False, ("summary",)
),
```

- `family`: group name (context_free, linear_contextual, logistic, bayesian, ensemble, hybrid, tree_ensemble, continuous)
- `needs_context`: whether the policy requires feature vectors
- `debug_views`: which debug pane builder to use

## 5. Add a Debug Pane

In `src/web/debug/context_free.py` (or a new builder file), create a builder:

```python
def build_my_debug_pane(snapshot: dict[str, Any]) -> ContextFreeDebugPane:
    return ContextFreeDebugPane(
        title="My Policy Debug",
        details=(("param", str(snapshot.get("param"))),),
    )
```

Register in `src/web/debug/__init__.py`.

## 6. Add Param Controls

In `src/web/ui/param_controls.py`, add a branch in `default_policy_param_controls()`:

```python
if policy_id == "my_policy":
    return (
        ParamControlSpec(
            key="param", label="My Param", control_type="slider",
            default_value=1.0, min_value=0.1, max_value=5.0, step=0.1,
            tooltip=ParamTooltip(
                title="My Param", intuition="Controls exploration.",
                formula="score = mean + param * bonus",
                tuning_hint="Increase for wider exploration.",
            ),
        ),
    )
```

## 7. Add to UI Policy Selector

In `src/web/main.py`, add to `_render_policy_selector()`:

```python
"my_policy": "My Policy",
```

## 8. Write Tests

```python
def test_my_policy_basic():
    policy = MyPolicy(param=1.0, seed=42)
    arms = ["a", "b", "c"]
    policy.reset()
    choice = policy.select_arm(context={"step": 1}, arms=arms)
    assert choice in arms
    policy.update(context={"step": 1}, arm=choice, reward=1.0)
    snap = policy.get_debug_snapshot()
    assert "param" in snap
```

## 9. Run Quality Gates

```bash
ruff check src/web/policies/my_policy.py
pytest tests/flet_redesign -q -k my_policy
```

## Tips

- Use `numpy` for matrix operations (contextual policies), plain Python for context-free.
- Always validate parameters in `__init__` — the factory passes user-provided params.
- Seed determinism is critical: same seed + same world must produce identical results.
