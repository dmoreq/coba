# Systematic Fix & Logic Review of Decision Trace Panel

Review the Decision Trace table logic to ensure all policy choices (UCB, Thompson, Softmax, Epsilon-Greedy) display accurate, mathematically correct selection reasons, fix the transient drift detection alarm logging bug, and implement a premium "explore vs. exploit" flag indicator.

---

## 1. Core Issues & Root Causes

### Issue A: Hardcoded Decision Reasons
Currently, `web/core/trace_builder.py` hardcodes the choice reason for any successful arm selection to `"highest_ucb"`:
```python
chosen_reason = "abstained" if abstained else ("highest_ucb" if chosen_arm else "none")
```
This is mathematically incorrect for non-UCB policies:
* **Thompson Sampling** (TS, LinTS, LogisticTS) chooses arms based on **posterior sample values**.
* **Softmax** chooses arms by sampling from a **softmax probability distribution**.
* **Epsilon-Greedy** either **explores randomly** (with probability epsilon) or **exploits greedily** (highest predicted mean).

### Issue B: Stagnant "Flags" Column & Missed Drift Alarms
The sequential drift detector (`PageHinkleyDetector`) triggers drift inside `bandit.update(reward)`. When drift is detected, it resets the arm's model and immediately calls `detector.reset()`, returning the drift state back to `False`.
In `web/core/simulator.py`, the `do_one_step` function builds the step's trace entry **before** calling `bandit.update()`. Because the trace is generated before the update, and because the drift state is transiently reset during the update, **drift alarms are never captured in the Decision Trace table**, leaving the "Flags" column completely empty for drift events!

---

## 2. Systematic Refactoring Plan

### 1. Dynamic Selection Reason Mapping (SOLID & OOP)
Expose the `policy_type` to `build_trace(...)` and map selection reasons dynamically based on the policy and score characteristics:
* **Thompson Sampling & Neural Linear**: `"posterior_sampling"`
* **Softmax**: `"softmax_sampling"`
* **Epsilon-Greedy**:
  * `"epsilon_exploration"` (when pulling an arm with a cold start/exploration score)
  * `"greedy_exploitation"` (when pulling the best predicted mean arm)
* **UCB Policies**: `"highest_ucb"`

### 2. Transient Drift Capture & Loop Sequence Order
* In `ClusterBandit.update(...)` (`src/coba/bandit.py`), record a transient boolean flag `self._drift_detected_last_step = True` whenever the Page-Hinkley detector signals drift.
* In `do_one_step` (`web/core/simulator.py`), execute the `bandit.update()` call **before** generating the trace entry, read & clear the `_drift_detected_last_step` flag, and feed the correct drift outcome directly into `build_trace(...)`.

### 3. Premium Explore/Exploit Indicators in Flags Column
Dynamically classify every decision step in the trace panel:
* Find the arm with the highest expected reward estimate (`mean` prediction).
* If the chosen arm has the highest expected reward, flag the step as `"★ EXPLOIT"`.
* If a different arm was pulled (due to UCB bonus, random sampling, or TS uncertainty), flag the step as `"✈ EXPLORE"`.
* Combine this with `"⚠ DRIFT"` and `"✋ ABSTAIN"` alarms.
* Style the rows in Dash DataTable conditionally to highlight exploration vs exploitation and drift warnings cleanly.

---

## User Review Required

> [!IMPORTANT]
> The loop reordering in `do_one_step` changes trace generation to execute post-update. All decision-time statistics (context, scores, chosen arm) are calculated *before* the update and preserved, guaranteeing that the logging metrics perfectly reflect the state of the model at the exact moment the decision was made.

---

## Proposed Changes

### Core Library (Bandit Policy)

#### [MODIFY] [bandit.py](file:///Users/quy.doan/Workspace/personal/coba/src/coba/bandit.py)
* Initialize `self._drift_detected_last_step = False` in `__init__`.
* Update `update` method to set `self._drift_detected_last_step = True` when Page-Hinkley triggers a drift event.

---

### Web App Simulator & Rendering

#### [MODIFY] [trace_builder.py](file:///Users/quy.doan/Workspace/personal/coba/web/core/trace_builder.py)
* Update `build_trace` signature to accept `policy_type: str | None = None`.
* Construct policy-appropriate `chosen_reason` strings based on the active policy type and score details.

#### [MODIFY] [simulator.py](file:///Users/quy.doan/Workspace/personal/coba/web/core/simulator.py)
* Update `do_one_step` loop order to perform `bandit.update(...)` first, capture the transient `_drift_detected_last_step` flag from the bandit, and then build the step trace entry.

#### [MODIFY] [trace_panel.py](file:///Users/quy.doan/Workspace/personal/coba/web/components/trace_panel.py)
* Implement dynamic `Flags` population: `"★ EXPLOIT"` when chosen arm matches the best mean estimate arm, `"✈ EXPLORE"` otherwise.
* Apply premium conditional styling in Dash DataTable to color-code exploit vs explore steps and highlight drift alarms.

---

## Verification Plan

### Automated Tests
* Run the test suite:
  ```bash
  pytest tests/ -v -p no:asyncio
  ```

### Manual Verification
* Start the Dash server.
* Select UCB1: Verify that steps are flagged with `"★ EXPLOIT"` or `"✈ EXPLORE"`, and the reason displays `"highest ucb"`.
* Select Thompson Sampling: Verify that reasons display `"posterior sampling"`.
* Select Drift Detection: Set a high drift rate, run the simulation, and verify that the row turns yellow with the `"⚠ DRIFT"` flag when drift occurs.
