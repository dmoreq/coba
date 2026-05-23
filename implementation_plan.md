# Systematic Fix & DRY/SOLID Refactoring of Bandit Lessons

Resolve the React-side rendering error `TypeError: e.map is not a function` in the Dash Mantine Components (DMC) library, fix the system-wide Decision Trace arm extraction bug, and refactor the lesson files to eliminate ~1,400 lines of redundant, duplicated callbacks by applying **SOLID** and **DRY** design principles.

---

## 1. Core Issues & Root Causes

### Issue A: Invalid Slider Marks (DMC v2 Bug) — [COMPLETED]
Under `dash-mantine-components>=2.0.0` (built on Mantine v7), the `marks` prop of `dmc.Slider` accepts a **list of dictionaries** (e.g. `[{"value": 0.0, "label": "0"}, ...]`) rather than a Python dictionary (e.g. `{0.0: "0", ...}`). A Python dictionary is serialized as a JSON object, causing the underlying React component's `.map()` operation to fail with:
`TypeError: e.map is not a function`

### Issue B: High Boilerplate & Code Duplication — [COMPLETED]
Across the 8 core interactive bandit lessons, there is a massive amount of code duplication (~1,400 lines). Each lesson duplicates the exact same 9 callbacks for:
1. `control_simulation` (stepping, running, pausing)
2. `set_speed` (speed multipliers)
3. `step_simulation` (stepping through simulation batches)
4. `update_arm_bar` (rendering arm score figures)
5. `update_pull_histogram` (rendering pull count figures)
6. `update_reward_chart` (cumulative rewards)
7. `update_regret_chart` (cumulative regret)
8. `update_trace_table` (rendering trace history)
9. `update_step_counter` (counter labels)

### Issue C: No Arm Selection in Decision Trace & Lack of Learning [NEW]
The simulator's `_extract_arm` helper expects the chosen arm to be under the attribute `"arm"`. However, the `BanditDecision` schema (from `src/coba/schemas.py`) stores the selected arm inside the attribute **`"chosen_arm"`**.
This causes the extractor to always return `None`, meaning:
* The decision trace table displays `"—"`.
* The `bandit.update()` call is bypassed, stopping the model from learning.
* Pull counts and arm scores remain stagnant at default levels.

---

## 2. Refactoring Design (DRY, SOLID, OOP)

To fix this systematically, we will apply the following design patterns:

### Single Responsibility & Separation of Concerns (SOLID: SRP)
* **Standard Boilerplate** -> Shared and encapsulated in `base_lesson.py`.
* **Lesson Modules** -> Handle only their specific layout, parameters, slider config cards, and custom visualization callbacks (e.g., Thompson's `beta-grid` or Cluster Routing's `cluster-scatter`).

### DRY (Don't Repeat Yourself)
* We will decouple the core simulation loop callbacks from the policy session creation callback.
* We will expose a new shared function `register_simulation_callbacks` in `base_lesson.py` that registers the 9 standard callbacks for any lesson module.

---

## 3. Detailed Proposed Changes

### [MODIFY] [simulator.py](file:///Users/quy.doan/Workspace/personal/coba/web/core/simulator.py)
* Refactor `_extract_arm` to check both `"chosen_arm"` and `"arm"` attributes and dictionary keys.
* This systematically fixes the decision trace and learning workflow across all 17 lessons.

```python
def _extract_arm(decision: Any) -> str | None:
    """Extract chosen arm from bandit decision, supporting both coba schemas and fallbacks."""
    if decision is None:
        return None
    if hasattr(decision, "chosen_arm"):
        return decision.chosen_arm
    if hasattr(decision, "arm"):
        return decision.arm
    if isinstance(decision, dict):
        return decision.get("chosen_arm") or decision.get("arm")
    return None
```

---

## 4. Verification Plan

### Automated Tests
1. Run `pytest` to verify the codebase structure is intact and no unit/smoke tests fail.
   ```bash
   .venv/bin/pytest
   ```
2. Verify code coverage meets the high standard required by global rules.
   ```bash
   .venv/bin/pytest --cov=. --cov-report=term-missing --cov-fail-under=90
   ```

### Manual Verification
* Start the Dash web server and visually confirm:
  * Decision Trace table displays correct chosen arms (e.g. "Variant A", "Variant B").
  * Run simulation and verify that pull counts, regret charts, and arm scores update dynamically as the bandit learns.
