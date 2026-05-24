# Web Module — Comprehensive Testing Plan

Last updated: 2026-05-24
Scope: `src/web/` directory (79 source files, 142 existing tests)
Audit basis: Deep analysis of all 29 test files against 79 source files

---

## 1) Executive Summary

The existing 142-test suite provides adequate **structural coverage** (most modules have at least one test) but critically lacks **mathematical precision verification**. Only 1 test (`test_contextual_math.py`) verifies exact formula outputs. 11 of 15 policies have no parameter validation tests. Zero debug snapshot tests verify values — they only check key existence. No shared test fixtures, base classes, or property-based testing infrastructure exists.

**This plan targets 142 → 280+ tests** with mathematical verification for all 15 policies, complete parameter validation coverage, debug snapshot value accuracy, deterministic replay tolerance, and comprehensive edge case coverage.

---

## 2) Test Infrastructure — New Architecture

### 2.1 Shared Fixtures (`conftest.py` — NEW)

```python
# tests/flet_redesign/conftest.py

@pytest.fixture(scope="session")
def seed_rng():
    """Return seeded RNG for deterministic test generation."""
    return random.Random(0)

@pytest.fixture
def clinic_world():
    """Fresh rural_clinic world with seed=0."""
    ...

@pytest.fixture(params=["rural_clinic", "moviematch", "newsfeed",
                        "shopsmart", "ridepilot", "gamebot", "labtrial"])
def any_world(request):
    """Parametrized fixture for all 7 worlds."""
    ...

@pytest.fixture(params=[...all 15 policy IDs...])
def any_policy_id(request):
    """Parametrized fixture for all 15 policy IDs."""
    ...

@pytest.fixture
def recorded_simulator():
    """Simulator that records internal policy state after each step."""
    ...
```

### 2.2 Base Test Classes (`test_harness.py` — NEW)

```python
class BasePolicyTest:
    """Shared test template for all 15 policies."""
    policy_cls: type
    valid_params: dict
    invalid_params: list[dict]

    def test_construct_rejects_invalid(self):
        """Every invalid param dict raises ValueError."""
        ...

    def test_reset_restores_determinism(self):
        """Same seed → same arm selection sequence."""
        ...

    def test_empty_arms_raises(self):
        """select_arm with empty list raises ValueError."""
        ...

class BaseDebugSnapshotTest:
    """Shared test template for debug snapshot value accuracy."""
    ...

class BaseWorldTest:
    """Shared test template for world reward model verification."""
    ...
```

### 2.3 Stub/Mock Library (`_stubs.py` — NEW)

Consolidate all duplicated stubs (DummyWorld, GreedyStubPolicy, BernoulliBanditWorld, PerfWorld, SimpleContinuousWorld) into one shared module to eliminate duplication across test files.

---

## 3) Phase 1: Deep Unit Testing (165 target tests total)

### 3.1 Mathematical Formula Verification (30 tests) — **CRITICAL**

**File:** `tests/flet_redesign/test_math_precision.py` (NEW)

| # | Policy | Formula Verified | Input | Expected Output |
|---|--------|-----------------|-------|----------------|
| 1 | UCB1 | `score = μ + α·√(2·log(N)/n)` | μ=0.4, n=10, N=50, α=1.0 | μ + √(2·ln(50)/10) ≈ 1.28 |
| 2 | UCB1 | Cold-start arms pulled exactly once each | 3 untried arms | each pulled once before UCB logic |
| 3 | Thompson | Beta(α+s, β+f) posterior parameters | s=3, f=1 | α'=4, β'=2 |
| 4 | Thompson | Mean of posterior | α=4, β=2 | 4/6 ≈ 0.667 |
| 5 | Softmax | `P(a) = exp(μ/τ)/Σexp` | μ=[0.5,0.3], τ=0.2 | probs sum to 1.0 |
| 6 | Softmax | Low τ → near-greedy | μ=[0.8,0.2], τ=0.01 | P(best) > 0.99 |
| 7 | GP-UCB | Welford variance after 3 updates | values=[0.1, 0.3, 0.5] | M2=(0.5-0.3)+(0.1-0.3)*(0.1-0.3) checked |
| 8 | GP-UCB | Uncertainty term | n=5, var=0.04 | √(0.04/6 + 1/6) |
| 9 | LinUCB | `θ = A⁻¹b` after 3 updates | known feature vectors | exact θ vector |
| 10 | LinUCB | Confidence bonus `α·√(xᵀA⁻¹x)` | known A⁻¹, x | exact bonus value |
| 11 | LinUCBSW | Window truncation | window=5, 8 updates | only last 5 used |
| 12 | LinUCBSW | A matrix after window shift | window=3, seq of 5 contexts | correct A from last 3 |
| 13 | LogisticUCB | Sigmoid output | x·θ=0.5 | 1/(1+e⁻⁰·⁵) ≈ 0.622 |
| 14 | LogisticUCB | Gradient update | r=1, p̂=0.4, η=0.1 | θ += 0.1·0.6·x |
| 15 | LinUCBHybrid | Shared A₀ + per-arm A_a decomposition | 2 arms, 2 shared features | correct split |
| 16 | LinUCBHybrid | Score = (θ₀+θ_a)ᵀx + bonus | known matrices | exact numerical score |
| 17 | TreeUCB | Bucket mean after 3 pulls | values=[0.2, 0.4, 0.6] | μ=0.4 |
| 18 | TreeUCB | UCB score in bucket | n=3, N=20, α=1.0 | μ + √(2·ln(20)/3) |
| 19 | TreeTS | Bucket posterior α,β | successes=4, failures=2 | α'=5, β'=3 |
| 20 | BootstrappedEnsemble | Head prediction aggregation | N=4, preds=[0.5,0.6,0.4,0.7] | mean=0.55, argmax |
| 21 | LinTS | θ̂ = A⁻¹b computation | known A,b | exact θ̂ vector |
| 22 | LinTS | Noise variance estimate | rewards=[0.1, 0.3, 0.5] | var of rewards |
| 23 | EpsilonGreedy | Mean reward after pulls | 3 arms, known rewards | exact means |
| 24 | EpsilonGreedy | Exploit when random < ε | ε=0, best=0.7, others=0.3 | picks best arm |
| 25 | Random | Uniform selection distribution | 3 arms, 1000 pulls | each ∼333 pulls (±50) |
| 26 | Logistic reward | `prob = sigmoid(logit + Σ wᵢ·fᵢ)` | known features, weights | exact probability |
| 27 | Logistic reward | Base rate 0.5 → logit=0 | base_rate=0.5, weight×feature=0.3 | sigmoid(0.3) |
| 28 | Feature normalization | Numeric scaling | [min,max]=[0,100], value=50 | 0.5 |
| 29 | Feature normalization | Categorical index division | categories=4, index=2 | 2/3 ≈ 0.667 |
| 30 | Feature normalization | Binary passthrough | value=1 | 1.0 |

### 3.2 Parameter Validation Coverage (22 tests) — **CRITICAL**

**Files:** Augment `test_contextual_policies.py`, `test_advanced_policies.py`, `test_continuous.py`

- LinUCB: alpha≤0, l2_lambda≤0, alpha=NaN, l2_lambda=-1
- LinUCBSW: window_size≤0, window_size="not-int"
- LinUCBHybrid: n_shared<1, alpha≤0
- LogisticUCB: alpha≤0, learning_rate≤0, learning_rate>1
- GPUCB: beta≤0, beta=NaN
- BootstrappedEnsemble: n_heads<2, n_heads=0
- TreeUCB: alpha≤0, context_key=""
- TreeTS: seed negative
- LinTS: prior_variance≤0, l2_lambda≤0
- CATS: exploration≤0, action_min≥action_max, action_min=NaN
- Plus: at least 3 invalid values per parameter (negative, zero, boundary, NaN where applicable)

### 3.3 Debug Snapshot Value Accuracy (15 tests) — **CRITICAL**

**File:** `tests/flet_redesign/test_debug_snapshot_accuracy.py` (NEW)

Each policy gets one test that:
1. Runs 10 deterministic steps
2. Calls `get_debug_snapshot()`
3. Verifies at least 3 numerical values against hand-computed expected values

| Policy | Values Checked |
|--------|---------------|
| UCB1 | per-arm mean, ucb_bonus, score = mean+bonus |
| Thompson | per-arm alpha_posterior = prior+successes, beta_posterior = prior+failures |
| Softmax | per-arm probabilities sum to 1.0 (±1e-9) |
| GP-UCB | per-arm mean via Welford, variance computed from M2 |
| LinUCB | A-trace = ΣA_ii, b-norm = ||b||₂, score=θᵀx+bonus |
| LinUCBSW | A matrix rebuilt from only window-bounded observations |
| LogisticUCB | theta norm after known update, sigmoid score |
| LinUCBHybrid | shared_theta norm, arm_theta norm, score decomposition |
| TreeUCB | bucket count, per-bucket mean |
| TreeTS | per-bucket alpha/beta posterior |
| BootstrappedEnsemble | per-head predictions, mean prediction |
| LinTS | theta_hat from A⁻¹b, theta_sample, noise_variance |
| Random | arm pull counts, pull distribution sum-to-1 |
| EpsilonGreedy | arm means matching reward_sum/pulls |
| CATS | best_action, best_reward, history_size |

### 3.4 State Machine Edge Cases (12 tests) — **HIGH**

| # | Edge Case | Steps | Expected |
|---|-----------|-------|----------|
| 1 | Reset mid-run | run 5 steps → reset → run 3 more | reset clears state, world+policy re-seeded |
| 2 | Play while running | set mode="running" → play() | mode unchanged, no crash |
| 3 | Pause while paused | set mode="paused" → pause() | mode unchanged, no crash |
| 4 | Step after horizon | horizon=5 → step×6 | step 6 raises or no-ops cleanly |
| 5 | Reset while running | mode="running" → reset() → mode="idle" | state cleared, mode idle |
| 6 | Step while running | play → step call while autoplaying | no double-step, no state corruption |
| 7 | Double reset | reset×2 | second reset is no-op or idempotent |
| 8 | Config change mid-run | run 10 steps → change seed → continue | old sim discarded, new sim starts fresh |
| 9 | World switch mid-run | run 10 → switch world → run 10 | new world, new sim, old state cleared |
| 10 | Speed change mid-autoplay | play → change speed 2x→8x | delay adjusts without stopping |
| 11 | Navigator rail double-click | fast double-click Home → Lesson | only one transition, no double-route |
| 12 | Lesson stage boundary | exactly meet objective min_steps=80, reward=40.00 | objective met exactly at threshold |

### 3.5 World Reward Model Correctness (10 tests) — **CRITICAL**

| # | Test |
|---|------|
| 1 | `expected_rewards()` returns correct probabilities for known context |
| 2 | Sigmoid stability near base_rate=0 (should return near-zero probability) |
| 3 | Sigmoid stability near base_rate=1 (should return near-one probability) |
| 4 | Numeric feature at min value → normalized to 0.0 |
| 5 | Numeric feature at max value → normalized to 1.0 |
| 6 | Categorical feature index 0 → normalized to 0.0 |
| 7 | Categorical feature last index → normalized to 1.0 |
| 8 | Binary feature 0 → float 0.0 |
| 9 | Binary feature 1 → float 1.0 |
| 10 | Rewards are only 0 or 1 (Bernoulli output) after 1000 samples |

### 3.6 Deterministic Replay Precision (5 tests) — **HIGH**

- Replace `==` with `math.isclose(a, b, rel_tol=1e-12)` for float comparisons
- Test replay payload roundtrip through JSON serialization
- Test that different seed → different first 3 step results (statistically)
- Test replay payload contains all required keys (config, steps)

### 3.7 Remaining Unit Tests (71 tests)

- TraceBuffer: CSV value verification, filter regex edge cases, large-trace (10k) performance, from_json with corrupted data, empty buffer edge cases
- Preferences: corrupted JSON handling, missing file, write permission error, concurrent access
- Checkpoint: corrupted JSON, missing state keys, empty trace, partial trace
- Curriculum: all 15 lessons have 5 stages, all policy→lesson mappings exist, objective boundary (exact equality at threshold), advance past stage 5 marks completed, completed→advance stays completed
- Arena: metrics value verification, arena_run_store commit order, multi-policy diff
- Comparison: CI95 formula verification (1.96·σ/√n), stats sorted by mean_reward descending, orchestrator empty policy list raises, world_id not found raises
- Router: route parameter extraction, nested route normalization
- Sandbox: invalid arm override rates (>1.0, <0.0, NaN), unknown world_id, unknown policy_id
- Shell: route stack dynamic updates
- Capabilities: all 15 policies have capability entries, family consistency
- PresetManager: corrupted file, empty file, missing file
- Checkpoint: save to non-writable path, load non-existent path

---

## 4) Phase 2: E2E Frontend Testing (25 tests)

Since Flet 0.85+ supports programmatic page control in test mode, we can write true E2E tests.

### 41. Flet Test Infrastructure

Flet 0.85+ provides `ft.app(test_mode=True)` which starts the app without opening a browser, allowing programmatic control via `page` object for E2E testing. We'll use `pytest-asyncio` for async WebSocket-based tests since Flet uses async WebSocket communication internally.

### 42. E2E Test Cases

| # | User Interaction | Verification |
|---|-----------------|-------------|
| 1 | Navigate to /lesson | Lesson view renders with theory card, stage indicator, treatment cards |
| 2 | Click Step button | Step counter increments, arm selection displayed, reward updated |
| 3 | Click Play button | Autoplay starts, step counter advances continuously |
| 4 | Click Pause during autoplay | Autoplay stops, step counter frozen |
| 5 | Click Reset | Simulator resets, step counter back to 0 |
| 6 | Change world dropdown | World switches, scene panel updates with new world title |
| 7 | Change policy dropdown | Policy switches, parameter controls update for new policy |
| 8 | Change speed dropdown | Autoplay speed changes (verify delay) |
| 9 | Click navigation rail: Arena | Arena route loads with metrics panel |
| 10 | Click navigation rail: Sandbox | Sandbox route loads with editor |
| 11 | Click navigation rail: Compare | Comparison route loads with policy picker |
| 12 | Lesson: complete stage 1 objective | Stage advances to 2, theory card updates |
| 13 | Lesson: complete all 5 stages | Lesson completion badge shown |
| 14 | Lesson: locked controls disabled | Stage 1 locks specific param controls |
| 15 | Arena: run 10 steps | Metrics panel shows reward/regret data |
| 16 | Arena: arm pull distribution | Pull counts shown after multiple steps |
| 17 | Sandbox: edit arm rate → validate | Validation passes or shows error |
| 18 | Sandbox: run scenario | Simulator runs with overrides, results shown |
| 19 | Compare: select policies → run | Side-by-side results table rendered |
| 20 | Compare: batch stats | Mean/std/CI95 shown for multiple seeds |
| 21 | Rapid Step clicks (10 in 1s) | No crashes, all steps processed correctly |
| 22 | Play → Pause → Play → Pause | Mode transitions work, no state corruption |
| 23 | Play → Reset during autoplay | Autoplay stops, simulator resets cleanly |
| 24 | Route change during autoplay | Autoplay stops when navigating away |
| 25 | Disconnect → reconnect | Preferences preserved, state fresh |

### 43. E2E Test Implementation

```python
# tests/flet_redesign/test_e2e.py

@pytest.mark.asyncio
async def test_step_button_increments_counter():
    page = await _create_test_page()
    assert "Step (0)" in _get_button_text(page)
    await _click_button(page, "Step")
    await asyncio.sleep(0.1)
    assert "Step (1)" in _get_button_text(page)
```

---

## 5) Phase 3: Edge Case Testing (14 tests)

### 5.1 Race Conditions (2 tests)

| # | Scenario | Verification |
|---|----------|-------------|
| 1 | Rapid successive Step clicks (50ms apart) | Each click produces exactly one step increment, no lost steps, no double-counting |
| 2 | Play → immediate Pause (10ms) | Autoplay starts and stops, state is "paused", step count ≤ 2 |

### 5.2 Invalid Constraints (4 tests)

| # | Scenario | Verification |
|---|----------|-------------|
| 1 | WorldConfig base_rate=1.5 | raises ValueError |
| 2 | WorldConfig base_rate=-0.3 | raises ValueError |
| 3 | ArmDef weights referencing non-existent feature | raises ValueError |
| 4 | FeatureDef with min > max | raises ValueError |

### 5.3 Extreme Inputs (5 tests)

| # | Scenario | Verification |
|---|----------|-------------|
| 1 | Context with float('inf') feature value | ConfigurableWorld raises or clamps |
| 2 | Context with None feature value | policy raises TypeError or handles gracefully |
| 3 | Zero-length feature_order for contextual policy | raises ValueError or handles gracefully |
| 4 | Context vector of all zeros | LinUCB selects arm (first-try behavior), no NaN |
| 5 | Horizon=0 run | simulator returns empty list or raises |

### 5.4 UI State (3 tests)

| # | Scenario | Verification |
|---|----------|-------------|
| 1 | Toggle dark mode | Charts/controls remain readable with dark background |
| 2 | Resize window to 400px wide | Layout doesn't overflow, no horizontal scroll |
| 3 | Long world description (>500 chars) | Text truncated or scrollable, no layout break |

---

## 6) Phase-to-File Mapping

| Phase | New Files | Modified Files | Test Count |
|-------|-----------|---------------|-----------|
| Infrastructure | `conftest.py`, `test_harness.py`, `_stubs.py` | — | 0 (setup) |
| Phase 1.1: Math Precision | `test_math_precision.py` | — | 30 |
| Phase 1.2: Param Validation | — | `test_contextual_policies.py`, `test_advanced_policies.py`, `test_continuous.py` | 22 |
| Phase 1.3: Debug Snapshot Accuracy | `test_debug_snapshot_accuracy.py` | — | 15 |
| Phase 1.4: State Machine Edges | — | `test_simulator.py`, `test_ui_layout.py` | 12 |
| Phase 1.5: World Reward Model | `test_world_reward_model.py` | — | 10 |
| Phase 1.6: Deterministic Replay | — | `test_phase1_regression.py`, `test_comparison.py` | 5 |
| Phase 1.7: Remaining Unit | `test_coverage_gaps.py` | multiple | 71 |
| Phase 2: E2E Frontend | `test_e2e.py` | — | 25 |
| Phase 3: Edge Cases | `test_edge_cases.py` | — | 14 |
| **TOTAL** | **8 new, 6 modified** | — | **~204 new (+142 existing = ~346 total)** |

---

## 7) Coding Standards

1. **DRY via base classes**: `BasePolicyTest` enforces consistent test patterns across all 15 policies.
2. **OOP via shared fixtures**: `conftest.py` provides parametrized fixtures instead of loop-based duplication.
3. **Stub consolidation**: `_stubs.py` eliminates 6 duplicated stub classes.
4. **Mathematical verification**: Every `assert` against formula output uses `pytest.approx` or `math.isclose` with explicit tolerance.
5. **Parametrize over loops**: `@pytest.mark.parametrize` for policy/world combinations, not `for pid in [...]`.
6. **Conventional commits**: One commit per phase:

```
git commit -F - <<'EOF'
test(web): add shared fixtures, base classes, and stub library

Add conftest.py with parametrized fixtures for all 15 policies
and 7 worlds. Add BasePolicyTest and BaseWorldTest base classes
in test_harness.py. Consolidate 6 duplicated stub classes into
_stubs.py.

Co-authored-by: CommandCodeBot <noreply@commandcode.ai>
EOF
```

---

## 8) Coverage Report

After all phases complete:

```bash
make coverage-web
# PYTHONPATH=src uv run pytest tests/flet_redesign -v -p no:asyncio \
#   --cov=src/web --cov-report=term-missing --cov-report=html
```

Target: **≥90% line coverage** on `src/web/`, with 100% branch coverage on all policy `select_arm`/`update` methods, world `sample_reward`/`expected_rewards`, and debug snapshot builders.

---

## 9) Immediate Next Actions

1. Create `tests/flet_redesign/conftest.py` with shared fixtures
2. Create `tests/flet_redesign/_stubs.py` with consolidated stubs
3. Create `tests/flet_redesign/test_harness.py` with base test classes
4. Create `tests/flet_redesign/test_math_precision.py` with 30 formula verification tests
5. Run: `make test-web`, verify all existing tests still pass
6. Run: `make coverage-web`, verify coverage baseline
