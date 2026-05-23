# Layout Redesign & Parameter Recovery Implementation Plan

This document provides a highly detailed, file-by-file blueprint for implementing the curriculum-wide layout redesign. It specifies the exact data contracts, callback structures, state-store mechanics, and mathematical formulations required to deliver custom ground-truth environments and parameter recovery charts across all 19 lessons.

---

## 1. Architectural Flow Diagram

```
[User adjusts Sliders] ──> [Recreate Session with Custom Ground Truth]
                                           │
                                           ▼
[Simulator (do_one_step)] ──> [Compute Custom Rewards & Extract Agent Estimates]
                                           │
                                           ▼
[Trace Store (Store Component)] ──> [Calculate Parameter Recovery (RMSE)]
                                           │
                                           ▼
                                [Plot Recovery Convergence]
```

---

## 2. File-by-File Code Modification Specifications

### A. Backend State Tracking: `web/core/session_service.py`
We will expand the standard initialization models to carry and make accessible the true configured coefficients.

*   **Modify `SessionStartRequest`**:
    Ensure `custom_weights` and `custom_base_rates` are accepted globally (instead of just on the arena page).
*   **Modify `_SessionRecord`**:
    Store the ground-truth coefficients so that callbacks can compare them with estimated weights:
    ```python
    @dataclass
    class _SessionRecord:
        bandit: ClusterBandit | ContinuousBandit
        lesson_id: str
        arms: list[str]
        n_features: int
        # ...
        custom_weights: dict[str, list[float]] | None = None
        custom_base_rates: dict[str, float] | None = None
    ```

---

### B. Estimation Extraction: `web/core/simulator.py`
We will implement defensive extraction logic to read the underlying model coefficients of different policy estimators at each step.

*   **Define `_extract_estimates(bandit, arms, n_features)`**:
    - For **Linear/Contextual Policies** (e.g., LinUCB, LinTS, Logistic UCB):
      Traverse the router's active arm models. Extract their fit weights (e.g., model's internal coefficient vector `model._coef` or mean vector `model._theta`).
    - For **Context-Free Policies** (e.g., UCB1, Thompson Sampling, Softmax):
      Extract empirical success rates (e.g., empirical averages $N_a / n_a$).
*   **Integrate into `do_one_step`**:
    Return `estimates` inside the trace entry:
    ```python
    trace_entry["arm_estimates"] = _extract_estimates(bandit, rec.arms, n_features)
    ```

---

### C. Controls & Sliders Interface: `web/components/controls.py`
We will build a modular sidebar controller to dynamically generate customization options.

*   **Implement `make_ground_truth_accordion(lesson_id, arms, n_features, is_contextual)`**:
    - Creates a `dmc.Accordion` with a single item `"Configure Hidden Ground Truth"`.
    - Labeled inputs:
      - Context-Free: Labeled `f"Arm {arm} Base Rate"` spanning a range $[0.0, 1.0]$.
      - Contextual: Labeled `f"Arm {arm} - Weight {feat}"` spanning a range $[-1.0, 1.0]$.

---

### D. Centralized UI Integration: `web/lessons/base_lesson.py`
The base lesson layout engine will be refactored to render the new controls and track estimation convergence automatically.

*   **Refactor `make_standard_layout`**:
    - Detect if the lesson has features (`lesson.n_features > 0`).
    - Embed the dynamic `make_ground_truth_accordion` inside the controls sidebar.
    - Add a new Plotly graph container `dcc.Graph(id=f"{lesson_id}-recovery-chart")` right next to the regret and reward charts.
*   **Update `register_simulation_callbacks`**:
    - Modify the `create_session` callback to gather all ground-truth slider values and pass them into the `SessionStartRequest`.
    - Register a new callback to update the convergence chart:
      ```python
      @callback(
          Output(f"{lesson_id}-recovery-chart", "figure"),
          Input(f"{lesson_id}-trace", "data"),
          State(f"{lesson_id}-session-id", "data")
      )
      def update_recovery_chart(trace, session_id):
          # Fetch ground-truth values from the session
          # Compute Euclidean Distance or Root Mean Squared Error (RMSE) at each step:
          # RMSE = sqrt( mean( (true_param - estimated_param) ** 2 ) )
          # Plot this series over time
      ```

---

## 3. Mathematical Specifications for Parameter Recovery

### A. Context-Free Policies (Empirical Success Rate Convergence)
For context-free arms $a \in A$, the error is the difference between the true Bernoulli success probability $p_a$ (custom base rate) and the agent's estimated success rate $\hat{p}_a$:

$$\text{Estimation Error}_t = \sqrt{\frac{1}{|A|} \sum_{a \in A} (p_a - \hat{p}_{a, t})^2}$$

---

### B. Contextual/Linear Policies (Regression Coefficient Alignment)
For linear arms, each arm is modeled by a $d$-dimensional coefficient vector $w_a$. The error is the root-mean-squared difference between the true weights and the agent's estimated regression parameters $\hat{\theta}_a$:

$$\text{Estimation Error}_t = \sqrt{\frac{1}{|A| \cdot d} \sum_{a \in A} \sum_{i=1}^{d} (w_{a, i} - \hat{\theta}_{a, i, t})^2}$$

As the number of interactions $t$ increases, both curves will asymptotically descend towards $0.0$, demonstrating successful model identification and proving parameter recovery.

---

## 4. Phased Execution Roadmap & Success Criteria

```
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Backend Hooks (Session & Estimation Extraction)               │
│ └─ Verify: Unit tests extract mock model parameters successfully.       │
├────────────────────────────────────────────────────────────────────────┤
│ Phase 2: Modular Ground-Truth Interface (controls.py & base_lesson.py) │
│ └─ Verify: Page loads correctly without any layout TypeErrors.         │
├────────────────────────────────────────────────────────────────────────┤
│ Phase 3: Live Convergence Charts (Plotly Curve Integrations)          │
│ └─ Verify: Running simulation plots real-time descending RMSE curves.  │
├────────────────────────────────────────────────────────────────────────┤
│ Phase 4: Regression & Curriculum-Wide Tests                            │
│ └─ Verify: make test-all is 100% green and error-free.                 │
└────────────────────────────────────────────────────────────────────────┘
```
