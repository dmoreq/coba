# Plan: Customizable Ground Truth & Multi-Algorithm Comparison Redesign

This document outlines the architectural plan to redesign the COBA simulation workspace. It introduces two core enhancements:
1. **Interactive Ground Truth Initialization**: Allow users to customize and preview the reward environment (weights, probabilities, and noise) before launching a simulation.
2. **Multi-Algorithm Comparison Dashboard**: Run and visualize multiple bandit policies side-by-side on the exact same context stream to compare their learning speed and asymptotic performance.

---

## 1. Interactive Ground Truth Initialization

Currently, reward functions (such as `linear_reward` and `categorical_reward`) are static and hardcoded inside `web/lessons/config.py`. To allow users to see "how well an algorithm learns," they must be able to specify the *target function* that the algorithm is trying to approximate.

### 1.1 Scope of Environment Customization
For any given lesson, the user should be able to open an **"Environment Configuration"** accordion/card before clicking "Run" or "Step".

We will support two core ground truth typologies:

#### A. Linear Contextual Environments (e.g., LinUCB, LinTS, NeuralLinear, SW-LinUCB)
*   **Coefficient Matrix ($\Theta$)**: For each arm $a$, users can configure the weight vector $\theta_a \in \mathbb{R}^d$ determining the expected reward $E[r | x, a] = x^T \theta_a$.
*   **Context Bias/Distribution**: Define whether contexts are sampled from a Uniform distribution, Standard Normal, or a skewed/high-correlation distribution.
*   **Observation Noise ($\sigma$)**: Additive Gaussian noise $e \sim \mathcal{N}(0, \sigma^2)$ applied to the expected reward.

#### B. Discrete/Context-Free Environments (e.g., UCB1, Thompson Sampling, Softmax)
*   **Arm CTR/Expected Rewards ($p_a$)**: Base conversion rates/rewards for each arm (e.g., `Arm A = 0.7`, `Arm B = 0.4`).
*   **Reward Type**: Bernouli (binary $0/1$ outcomes) vs. Gaussian (continuous rewards with custom variance).

### 1.2 User Interface & Interactive Visualizations

We will replace static layouts with a modular sidebar/collapsible accordion section containing:
1.  **Preset Configurations**: A dropdown containing standard templates:
    *   *Adversarial (Highly Confusing)*: Arms have very close expected rewards (e.g. $0.51$ vs $0.49$), making exploration critical.
    *   *Sparse Features*: Only a single feature dimension carries actual reward weights, testing the policy's feature selection speed.
    *   *Easy Contrast*: One arm is strictly dominant across all contexts.
2.  **Manual Weight Adjustments**: Dynamic slider inputs generated per arm and per feature. For example, if $K=3$ arms and $d=4$ features, we render a grid of sliders allowing the user to set $\Theta_{a, i} \in [-1.0, 1.0]$.
3.  **The "Ground Truth Surface Plot" (2D/3D Preview)**:
    *   For $d=1$ or $d=2$ feature dimensions, we will render a Plotly line/surface plot of $x^T \theta_a$ for each arm across the feature range.
    *   This lets the user visually identify the **optimal decision boundaries** (where the arm line colors cross) *before* the simulation runs.

### 1.3 Backend Architecture & Data Schema Changes

To support custom reward functions without rebuilding the whole registry, we will extend the session request models.

#### A. Session Registry & Configurations (`web/core/session_service.py`)
We will add optional fields to `SessionStartRequest` to carry user-defined reward schemas:

```python
class CustomRewardWeight(NamedTuple):
    arm: str
    weights: list[float]

class SessionStartRequest(NamedTuple):
    lesson_id: str
    arms: list[str]
    n_features: int
    config: BanditConfigModel
    # Added ground truth initializers:
    custom_weights: list[CustomRewardWeight] | None = None
    custom_base_rates: dict[str, float] | None = None
    noise_variance: float = 0.05
```

#### B. Dynamic Reward Construction (`web/core/simulator.py`)
In the session's step-loop, we construct the reward function dynamically based on the session's stored record instead of hardcoding `REWARD_FN`:

```python
def make_custom_reward_fn(rec: _SessionRecord) -> RewardFn:
    """Build a reward callable bound to the session's configured ground truth."""
    if rec.custom_weights:
        # Re-create a linear reward function with the user's customized matrix
        weights_dict = {item.arm: item.weights for item in rec.custom_weights}
        return linear_reward(weights_dict, noise=rec.noise_variance)
    elif rec.custom_base_rates:
        # Re-create a categorical reward function with base rates
        return categorical_reward(rec.custom_base_rates, noise=rec.noise_variance)
    return get_lesson_by_slug(rec.lesson_id).reward_fn
```

---

## 2. Multi-Algorithm Comparison Dashboard

To understand the practical tradeoffs of different exploration heuristics (e.g., UCB's deterministic confidence bounds vs. Thompson Sampling's posterior sampling), users must be able to run them side-by-side on the exact same workload.

### 2.1 The Core Scientific Constraint: Synchronized Context Streams

For a bandit comparison to be mathematically rigorous:
*   All compared algorithms **MUST** receive the *exact same stream of contexts* $\{x_t\}_{t=1}^T$.
*   If Policy A receives a favorable context sequence while Policy B receives a noisy, ambiguous sequence, the comparison is corrupted by selection bias.
*   **The Synchronized Loop**:
    1. The simulator draws a single context $x_t \sim D$.
    2. The simulator queries each active policy $i \in \{1 \dots M\}$: $a_{i, t} \leftarrow \text{decide}(x_t)$.
    3. The simulator evaluates the reward for each selected action independently under the *same* ground truth function: $r_{i, t} \sim P(r | x_t, a_{i, t})$.
    4. Each policy is updated with its *own* decision and reward: $\text{update}(x_t, a_{i, t}, r_{i, t})$.

### 2.2 Backend Architecture: `MultiAlgorithmSession`

We will extend `BanditSessionService` to support multi-bandit simulation state tracking.

```python
@dataclass
class _MultiSessionRecord:
    session_id: str
    bandits: dict[str, ClusterBandit | ContinuousBandit]  # policy_slug -> bandit instance
    arms: list[str]
    n_features: int
    step: int = 0
    created_at: float = field(default_factory=time.time)
    last_context: np.ndarray | None = None
    # Separate historical traces for each compared algorithm
    traces: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
```

During execution, `do_one_step` will execute a synchronized broadcast step:

```python
def do_one_step_multi(
    svc: BanditSessionService,
    session_id: str,
    reward_fn: RewardFn,
    n_features: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Step all compared policies concurrently using a shared context."""
    rec = svc.get_multi_record(session_id)
    step_num = rec.step + 1
    rec.step = step_num

    # 1. Generate one shared context
    context = np.random.normal(0, 1, size=n_features)
    rec.last_context = context

    step_results = {}

    # 2. Iterate through each active policy
    for name, bandit in rec.bandits.items():
        # Query decision
        decision = bandit.decide(context)
        chosen_arm = _extract_arm(decision)

        # Observe reward
        reward = reward_fn(chosen_arm, context)

        # Update policy
        bandit.update(context=context, arm=chosen_arm, reward=reward, propensity=1.0)

        # Build individual trace entry
        trace_entry = build_trace(
            step=step_num,
            context_raw=context,
            # ... scoring and stats ...
        )
        rec.traces[name].append(trace_entry)

    return step_results
```

### 2.3 UI/UX Redesign: The Multi-Agent Arena

We will build a dedicated, flagship dashboard: **"Multi-Algorithm Arena"** (`web/lessons/algorithm_arena.py`).

#### A. Sidebar Control Panel
*   **Policy Selection Checklist**: A list of checkboxes (or `dmc.MultiSelect`) allowing users to choose up to 4 algorithms to run simultaneously:
    *   `[x] LinUCB (α=1.0)`
    *   `[x] LinTS (v²=0.5)`
    *   `[x] Epsilon-Greedy (ε=0.1)`
    *   `[ ] Random Forest UCB (α=1.0)`
*   **Environment Preset Configuration**: Select the Ground Truth matrix/base-rates.
*   **Speed & Steps Controller**: Standard run, pause, single-step, and reset buttons applying to all agents simultaneously.

#### B. Comparative Real-Time Charts (The Plotly Dashboard)
1.  **Cumulative Regret Comparison (The Gold Standard)**:
    *   Line chart plotting cumulative regret $R_T = \sum_{t=1}^T \left(\max_a E[r_t|x_t, a] - E[r_t|x_t, a_{\text{chosen}}]\right)$ over time.
    *   Render 2-4 lines concurrently, colored by policy.
    *   Users will visually see the **concave curve** of fast-converging algorithms vs. the **linear slope** of sub-optimal algorithms.
2.  **Rolling Average Reward (Sliding Window)**:
    *   Shows a 50-step moving average of rewards.
    *   Helps users see how quickly each algorithm identifies the optimal decision boundaries.
3.  **Arm Selection Distributions**:
    *   Grouped bar chart showing what percentage of times each policy has pulled each arm. Optimal algorithms will quickly skew 90%+ towards the best arms.

---

## 3. Step-by-Step Implementation Plan

To ensure a smooth transition with zero regressions, we divide the refactoring into four isolated phases.

```
┌────────────────────────────────────────────────────────┐
│ Phase 1: Backend Foundations & Multi-Bandit Sessions   │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│ Phase 2: Interactive Ground Truth Controls & Preview   │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│ Phase 3: The Flagship "Multi-Algorithm Arena" Page     │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│ Phase 4: Verification, Type-Safety & Linter Approval   │
└────────────────────────────────────────────────────────┘
```

### Phase 1: Backend Foundations & Multi-Bandit Sessions
*   **Task 1**: Update `BanditSessionService` in `web/core/session_service.py` to support `_MultiSessionRecord` creation and lookup.
*   **Task 2**: Implement `do_one_step_multi` inside `web/core/simulator.py` to broadcast shared contexts to multiple bandits.
*   **Task 3**: Write robust unit tests verifying context synchronization and independent state updates across compared estimators in `web/tests/unit/test_simulator.py`.

### Phase 2: Interactive Ground Truth Controls & Preview
*   **Task 1**: Refactor `linear_reward` and `categorical_reward` builders to accept custom weight matrices.
*   **Task 2**: Create a reusable UI component (`components/ground_truth_editor.py`) rendering weight sliders and preset configurations.
*   **Task 3**: Build a Plotly preview figure displaying expected reward surfaces/functions before simulation launch.

### Phase 3: The Flagship "Multi-Algorithm Arena" Page
*   **Task 1**: Create `web/lessons/algorithm_arena.py` containing the side-by-side comparative layout.
*   **Task 2**: Map comparative callbacks (`update_comparison_charts`, `do_multi_step`).
*   **Task 3**: Register the new page as Lesson 19 in `web/lessons/config.py` and `web/lessons/registry.py`.

### Phase 4: Verification, Type-Safety & Linter Approval
*   **Task 1**: Run the full workspace unit/integration test suite (`make test-all`).
*   **Task 2**: Verify static typing is fully clean (`make check-types check-types-web`).
*   **Task 3**: Verify code style compliance (`make lint`, `make format`).

---

## 4. Technical Challenges & Design Tradeoffs

### 4.1 Memory Allocation & Session State
*   **Problem**: Storing historical traces for up to 4 simultaneous scikit-learn models (especially Random Forest estimators or Neural Network embeddings) within a stateful session can cause backend memory bloat if sessions are not cleaned up.
*   **Solution**: Keep the standard 15-minute expiration/eviction schedule active (`evict_stale_sessions`). Ensure that trace records are capped strictly to the last 200 items, and only incremental summary matrices are kept.

### 4.2 UI Latency under High Speeds
*   **Problem**: When simulating 4 policies in parallel at 30 steps/sec, executing Dash state transfers for multiple large Plotly charts can bottleneck the browser rendering loop.
*   **Solution**: Batch callbacks. Under high speeds, accumulate simulation steps in the background and only trigger a Dash graph update every 10 or 20 steps (similar to the existing batch-size speed scaling in `base_lesson.py`).
