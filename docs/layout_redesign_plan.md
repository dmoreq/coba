# Plan: Curriculum-Wide Lesson Redesign & Parameter Recovery Tracking

This document outlines the design and implementation plan to update **all standard contextual and context-free lessons** across the COBA platform.

The goal of this redesign is twofold:
1. **Interactive Ground Truth Configuration**: Allow the user to manually configure the ground truth environment (base success rates for context-free lessons, linear coefficients/weights for contextual ones) before starting the simulation.
2. **Ground Truth Recovery Diagnostics**: Introduce an **Estimation Convergence/Parameter Recovery** panel that visualizes how close the bandit's estimated parameters ($\hat{\theta}$ or $\hat{p}$) are to the user-defined ground truth as learning progresses.

---

## 1. Architectural Strategy: Reusable & Modular Base Components

Rather than rewriting all 19 lesson modules individually, we will implement this feature elegantly by updating the centralized lesson engine in **`web/lessons/base_lesson.py`**, **`web/components/controls.py`**, and **`web/core/session_service.py`**. This guarantees consistent UI, 100% backwards-compatibility, and robust testability.

### Key Changes
1. **`web/core/session_service.py`**:
   - The session initialization will be updated to accept dynamic configurations of `custom_weights` or `custom_base_rates` across all lessons.
   - Extend `_SessionRecord` to track and report true model coefficients alongside the agent's estimated weights/means at each step.

2. **`web/components/controls.py` / `web/lessons/base_lesson.py`**:
   - Introduce a collapsible **"Configure Environment Ground Truth"** panel below the standard control widgets.
   - For **Context-Free lessons** (e.g., UCB1, Softmax, Thompson Sampling): Render sliders for each arm's base reward success rate ($p_a \in [0, 1]$).
   - For **Contextual/Linear lessons** (e.g., LinUCB, LinTS, Hybrid): Render grid columns of sliders for linear weights ($w_{a, f} \in [-1, 1]$).

3. **New Chart Component: "Ground Truth vs. Estimates"**:
   - Introduce a new interactive Plotly tracking chart: **Parameter Estimation Error / Distance**.
   - For context-free models, plot the true success rates $p$ side-by-side with the agent's estimated averages $\hat{p}$.
   - For linear models, plot the parameter recovery metric over time (e.g., Mean Squared Error or Cosine Similarity between true weight vector $w_a$ and estimated model weights $\hat{\theta}_a$):
     $$\text{Estimation Error}_t = \frac{1}{|A|} \sum_{a \in A} \|w_a - \hat{\theta}_{a, t}\|_2^2$$

---

## 2. Detailed Phased Implementation Plan

```
1. Update Backend Session & Simulators -> verify: Unit tests pass for customizable configs
2. Create Ground Truth Sliders UI    -> verify: Sliders correctly update session start requests
3. Design Parameter Recovery Chart   -> verify: Live chart shows shrinking estimation error
4. Roll Out globally to all Lessons   -> verify: Integration and smoke-tests are fully green
```

### Phase 1: Backend Session & Simulator Adjustments
- Modify `default_session_request` to gather slider inputs from the UI.
- Wire these slider configurations directly into `SessionStartRequest(custom_weights=..., custom_base_rates=...)`.
- Update `do_one_step` in `web/core/simulator.py` to:
  - Generate rewards based on user-defined custom base rates or weights.
  - Return the agent's estimated coefficients or arm posterior values (e.g., $\hat{\theta}_a$, $\hat{\mu}_a$, or empirical means $\hat{p}_a$) inside `extra_data` or `arm_scores` so the chart can fetch them.

### Phase 2: Dynamic UI Sliders in the Control Shell
- Create a reusable helper component `make_ground_truth_controls(lesson_id, arms, n_features, is_contextual)`:
  - Supports discrete/context-free sliders (labeled with success probability).
  - Supports feature-specific contextual weights (labeled per coordinate).
- Embed this collapsible accordion panel directly inside `make_standard_layout` in `web/lessons/base_lesson.py`.

### Phase 3: Plotly Parameter Recovery Visualizations
- Add a new chart container in `make_standard_layout` next to Cumulative Regret:
  - **"Learning & Recovery Convergence Chart"**
- For Context-Free lessons:
  - Render a side-by-side bar chart of **True Rate** vs. **Estimated Rate** $\hat{\mu}$.
- For Contextual lessons:
  - Render a line chart tracking the **Average Root Mean Squared Error (RMSE)** of the weight vectors at each step. As the agent pulls arms and obtains observations, this error curve should steadily decline towards $0.0$, proving successful learning.

### Phase 4: Curricula Verification & Verification Suite
- Register dynamic callbacks in `base_lesson.py` to recreate the session whenever the ground truth sliders are modified.
- Write extensive test cases verifying:
  - Dynamic slider values correctly feed into the simulation.
  - Estimated parameters are correctly exposed by the various arm model subclasses.
  - All 19 lessons render, load, and simulate cleanly under custom environments.

---

## 3. Mockup of the Redesigned Dashboard Layout

```
+─────────────────────────────────────────────────────────────────────────────+
│                     LESSON HEADER (e.g., LinUCB Contextual)                  │
+─────────────────────────────────────────────────────────────────────────────+
│  THEORY CARD & EXPLANATIONS          │  SIMULATION CONTROLS & LOGS          │
│  "Choose your policy parameters..."  │  - Speed: [ 1 | 5 | 15 | 30 ] steps/s│
│                                      │  - Action Buttons: [ Run | Pause ]   │
│                                      │                                      │
│                                      │  +─ [v] CONFIGURE CUSTOM GROUND TRUTH──+
│                                      │  │ Arm A Weight 1: [---o---] 0.5     │
│                                      │  │ Arm A Weight 2: [---o---] -0.3    │
│                                      │  │ Arm B Weight 1: [---o---] -0.2    │
│                                      │  +───────────────────────────────────+
+──────────────────────────────────────┴──────────────────────────────────────+
│                      LIVE PERFORMANCE & LEARNING DIAGNOSTICS                │
│                                                                             │
│  +───────────────────────────────+   +───────────────────────────────────+  │
│  │   CUMULATIVE REGRET (Regret)  │   │  PARAMETER RECOVERY ERROR (RMSE)  │  │
│  │  Lower is better, flattens   │   │  Error decreases to 0.0 as        │  │
│  │  as policy learns.           │   │  learning converges.              │  │
│  │  ~~~~~~~~~~~\                │   │  ~~~~~~~~~~~\                     │  │
│  │              \___________     │   │              \                    │  │
│  │                           \__ │   │               \________           │  │
│  +───────────────────────────────+   +───────────────────────────────────+  │
+─────────────────────────────────────────────────────────────────────────────+
```

---

## 4. Risks & Mitigations

- **Complexity in Arbitrary Estimators**: Some models (like neural linear embeddings or random forests) do not expose simple linear weights.
  - *Mitigation*: For models without explicit linear weights, track parameter recovery using **empirical decision accuracy** or **cosine similarity of arm scores** rather than regression coefficient MSE.
- **Performance Overhead**: Updating high-frequency interval steps with large coordinate matrices could cause lag.
  - *Mitigation*: Only recalculate and update the Plotly Parameter Recovery chart every 10 steps (using a modulo step check), keeping the UI incredibly fast and fluid.
