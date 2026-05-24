# Web Module — Comprehensive Implementation Plan

Last updated: 2026-05-24
Source: Deep analysis of `src/web` (66 files, 7 layers, 50 unfinished items)
Tracker companion: `docs/flet_redesign_progress_tracker.md`

---

## 1) Purpose

This plan covers every unfinished piece of work identified in the `src/web` analysis (50 items), organized into 10 sequenced phases with concrete deliverables, file-level targets, and acceptance criteria. Each phase builds on the last — no phase ships in isolation.

---

## 2) Delivery Principles

1. **Interactive core first.** The app must run simulations before features matter.
2. **Backend-then-UI.** Where backend logic exists, prioritize wiring it to pages.
3. **Test-as-you-go.** Every phase includes its own test targets — no test debt accumulation.
4. **Complete the matrix.** Missing algorithms, worlds, and lessons are tracked explicitly.
5. **Unify before polish.** Sim loop unification happens before chart/table rendering.

---

## 3) Phase 1: Interactive Core Wiring (6 items)

**Priority:** Critical — the app renders but cannot run a single simulation step.

### Objectives
Wire `DiscreteSimulator`, `RunController`, `PreferencesStore`, and `LessonProgressState` into the Flet shell so users can run, pause, reset, and advance through lessons with live state reflected in the UI.

### Deliverables

#### 1.1 — Wire DiscreteSimulator to Play/Pause/Step/Reset buttons

- File: `src/web/main.py`
- Add a `_create_simulator(world_id, policy_id, seed)` factory called on route entry.
- Attach `on_click` callbacks to Play, Pause, Step, Reset buttons.
- Play starts a repeating timer (from `UserPreferences.speed`) that calls `simulator.step()`.
- Step runs one step and pauses.
- Pause stops the timer.
- Reset calls `simulator.reset()` and clears the view.

#### 1.2 — Wire live state to UI

- File: `src/web/main.py`
- After each `simulator.step()`, rebuild the view from current `SimulationState` and `TraceBuffer`.
- Call `page.update()` after each step.
- The three-pane layout must show: current context (left), chosen arm + reward (center), trace rows and running metrics (right).

#### 1.3 — Wire RunController to simulator

- File: `src/web/main.py`
- Instantiate `RunController()` at page scope.
- Dispatch `RunController.play()`, `RunController.pause()`, `RunController.step()`, `RunController.reset()` from button callbacks.
- Guard against double-play, step-while-running, etc.

#### 1.4 — Connect PreferencesStore to active state

- Files: `src/web/main.py`, `src/web/ui/preferences.py`
- On route change, read `PreferencesStore.load()` for `world_id`, `policy_id`, `speed`.
- On every preference change (world dropdown, policy picker, speed slider), call `PreferencesStore.save()`.
- On disconnect (`on_disconnect`), persist current preferences.

#### 1.5 — Connect lesson stage advancement

- Files: `src/web/main.py`, `src/web/curriculum/lessons.py`
- On lesson route entry, instantiate `LessonProgressState` from the active lesson.
- After each step, call `evaluate_lesson_objective()` — if satisfied, call `progress.advance()`.
- Render the current stage's `render_theory_stage_markdown()`.
- Show a "Stage Complete / Advance" prompt when objectives are met.

#### 1.6 — Wire ParamControlSpec to live parameter updates

- Files: `src/web/main.py`, `src/web/ui/param_controls.py`
- Render each `ParamControlSpec` as a Flet control: `ft.Slider` for numeric, `ft.Dropdown` for options.
- On value change, rebuild the policy via `build_policy()` with updated params, reset the simulator.
- Honor `locked_control_keys_for_stage()` — disabled controls during locked stages.

### Exit Criteria
- User can select world + policy, press Play, and watch live step-by-step updates.
- Step, Pause, Reset all work.
- Preferences persist across route changes and app restart.
- Lesson stages advance when objectives are met.
- Parameter changes reset the simulator with new policy.

---

## 4) Phase 2: Missing UI Pages (5 items)

**Priority:** High — comparison and sandbox backends exist but have zero UI surface.

### Objectives
Create `ui/pages/` directory with comparison and sandbox pages, wire them into routing, and build reusable components for diff viewing and stats display.

### Deliverables

#### 2.1 — Create `ui/pages/` directory structure

- Files: `src/web/ui/pages/__init__.py`

#### 2.2 — Build comparison page

- File: `src/web/ui/pages/comparison_page.py`
- UI: policy multi-select, world picker, seed input, horizon slider, Run button.
- Results: side-by-side table of `ComparisonRunResult` per policy.
- Charts: overlaid cumulative reward lines per policy.
- Use `comparison/orchestrator.py` → `run_policy_comparison()`.
- Wire into `main.py` via a new `build_comparison_ui_model()` in `view_models.py`.

#### 2.3 — Build sandbox page

- File: `src/web/ui/pages/sandbox_page.py`
- UI: world picker, policy picker, arm base rate overrides (sliders), horizon, seed.
- Instantiate `SandboxEditor`, call `set_param()` on changes, validate with `build_world_override()`.
- Run button → create `ConfigurableWorld` with overrides → `DiscreteSimulator` → display results.
- Guard against invalid configs (negative rates, rates > 1.0, etc.).

#### 2.4 — Build snapshot diff view component

- File: `src/web/ui/components/snapshot_diff_view.py`
- Takes two `SnapshotDiffResult` objects, renders before/after side-by-side.
- Highlights changed keys with color coding.
- Used in comparison page (compare two runs) and arena (compare current/previous).

#### 2.5 — Build batch summary stats panel

- File: `src/web/ui/components/batch_summary_panel.py`
- Takes `list[PolicySummaryStats]`, renders table with: policy_id, n_runs, mean_reward ± ci95, mean_regret.
- Sortable by column.
- Used in comparison page for multi-seed batch runs.

### Exit Criteria
- Comparison page: select 2+ policies → run → see results table + overlaid charts.
- Sandbox page: edit arm rates → validate → run → see results.
- Diff viewer highlights changed fields.
- Stats panel shows mean/std/CI95.

---

## 5) Phase 3: Missing Algorithms (3 items)

**Priority:** Medium — LinTS is a curriculum dependency. Drift detectors complete the algorithm matrix.

### Objectives
Implement LinTS, Drift Detector A, and Drift Detector B with debug snapshot support and unit tests.

### Deliverables

#### 3.1 — Implement LinTS (Linear Thompson Sampling)

- File: `src/web/policies/lints_policy.py`
- Implements `BanditPolicy[str, dict[str, Any]]` + `DebugSnapshotProvider`.
- Posterior sampling from `N(θ̂, v²·A⁻¹)` where θ̂ = A⁻¹b, v² = estimated variance.
- Parameters: `feature_order`, `prior_variance` (default 1.0), `l2_lambda`.
- `get_debug_snapshot()` returns: feature_order, A matrices, b vectors, sampled theta per arm, scores.
- Register in `policy_capabilities.py` as family=`linear_contextual`, `needs_context=True`.
- Register in `policy_factory.py` with id `"lints"`.
- Add debug pane builder in `debug/contextual.py`: `build_lints_debug_pane()`.

#### 3.2 — Implement Drift Detector A (CUSUM-based)

- File: `src/web/drift/cusum_detector.py` (new `drift/` subpackage)
- CUSUM detector with configurable threshold and drift magnitude.
- Wraps or implements CUSUM logic: accumulate deviations, trigger when threshold exceeded.
- Expose via `drift/__init__.py`: `CUSUMDriftDetector`.

#### 3.3 — Implement Drift Detector B (ADWIN-based)

- File: `src/web/drift/adwin_detector.py`
- ADWIN sliding window detector with adaptive window sizing.
- Expose via `drift/__init__.py`: `ADWINDriftDetector`.

#### 3.4 — Register both in algorithm tracker

- Update `policy_capabilities.py` (or create `drift_capabilities.py`).
- Update `docs/flet_redesign_progress_tracker.md` algorithm completion table.

### Exit Criteria
- LinTS passes convergence test against synthetic linear fixture.
- LinTS debug snapshot contains A, b, theta, scores per arm.
- CUSUM detector correctly identifies known drift points in fixture data.
- ADWIN detector correctly identifies known drift points in fixture data.
- All three have unit tests (target: +6 tests).

---

## 6) Phase 4: Debugger Completion (2 items + depth)

**Priority:** Medium — debugger parity principle requires all policies to have debugger support.

### Objectives
Add `get_debug_snapshot()` to all 5 context-free policies, build their debug pane builders, and deepen existing debug panes to surface the data already in snapshots.

### Deliverables

#### 4.1 — Add debug snapshots to context-free policies

- Files: `random_policy.py`, `epsilon_greedy_policy.py`, `ucb1_policy.py`, `thompson_policy.py`, `softmax_policy.py`
- Each gets `get_debug_snapshot()` returning: arm pull counts, per-arm mean reward, per-arm score/last_score, policy-specific internals (epsilon, tau, alpha/beta posteriors, etc.).
- Each class adds `DebugSnapshotProvider` to its inheritance.

#### 4.2 — Build context-free debug pane builders

- File: `src/web/debug/context_free.py` (new)
- `build_cf_debug_pane(snapshot)` returns `AdvancedDebugPane` with: arm pull distribution, best arm, reward means, cumulative regret.
- Shared builder — all 5 context-free policies use the same schema.

#### 4.3 — Deepen existing debug panes

- Files: `debug/contextual.py`, `debug/advanced.py`, `debug/continuous.py`
- `build_linucb_debug_pane`: surface A matrix trace/det, b vector, theta per arm, confidence bonus term per arm.
- `build_logistic_debug_pane`: surface gradient norms, learning rate, sigmoid scores.
- `build_gp_debug_pane`: surface per-arm mean, variance, uncertainty term.
- `build_ensemble_debug_pane`: surface per-head predictions, agreement ratio, variance of predictions.
- `build_tree_debug_pane`: surface bucket counts, per-bucket stats.
- `build_hybrid_debug_pane`: surface shared vs. arm-specific decomposition.
- `build_continuous_debug_pane`: surface action distribution samples, utility landscape, exploration radius.

#### 4.4 — Call debug snapshots during simulation

- Files: `src/web/main.py`, `src/web/simulator.py`
- After each step, if policy implements `DebugSnapshotProvider`, call `get_debug_snapshot()` and store in step metadata.
- Route debug pane data to the right panel alongside metrics.

### Exit Criteria
- All 14 policies implement `DebugSnapshotProvider`.
- Debug pane builders exist for all 8 policy families.
- Debug data updates live during simulation.
- Debug pane tests verify snapshot-to-pane transformation.

---

## 7) Phase 5: Missing Worlds (4 items)

**Priority:** Low — not blocking for core release, but required for 7/7 world target.

### Objectives
Implement ShopSmart, RidePilot, GameBot, and LabTrial with configs, presets, and world tests.

### Deliverables

#### 5.1 — ShopSmart world

- File: added to `src/web/worlds/core_worlds.py`
- Theme: e-commerce product ranking. Features: price_sensitivity (numeric), loyalty_tier (categorical), mobile_user (binary).
- 3 arms: discount_banner, premium_placement, social_proof.
- Difficulty: easy.

#### 5.2 — RidePilot world

- File: added to `src/web/worlds/core_worlds.py`
- Theme: ride-hailing dispatch. Features: surge_multiplier (numeric), trip_distance (numeric), time_of_day (categorical).
- 3 arms: standard_dispatch, priority_routing, pool_match.
- Difficulty: medium.

#### 5.3 — GameBot world

- File: added to `src/web/worlds/core_worlds.py`
- Theme: game difficulty adaptation. Features: player_skill (numeric), session_count (numeric), device_type (categorical).
- 3 arms: easy_mode, normal_mode, hard_mode.
- Difficulty: medium.

#### 5.4 — LabTrial world

- File: added to `src/web/worlds/core_worlds.py`
- Theme: clinical trial arm allocation. Features: biomarker_level (numeric), prior_response (binary), risk_group (categorical).
- 3 arms: control, low_dose, high_dose.
- Difficulty: hard.

#### 5.5 — Presets and registry

- Add 2 difficulty presets per world.
- Register in `CORE_WORLD_CONFIGS`.
- Add fixtures for deterministic testing.

### Exit Criteria
- 7/7 worlds exist in registry.
- Each world has ≥2 validated presets.
- World switching works for all 7 without restart.
- Each world has schema validation tests and fixture tests.

---

## 8) Phase 6: Missing Lessons (7 items)

**Priority:** Medium — curriculum coverage is 8/14 policies, need to close the gap.

### Objectives
Create lessons for the 6 non-context-free advanced policies (gp_ucb, bootstrapped_ensemble, linucb_hybrid, tree_ucb, tree_ts, cats) plus at least one lesson pairing an existing policy with a new world.

### Deliverables

#### 6.1 — Lesson: GP-UCB Exploration

- `lesson_id`: `"lesson_gp_ucb"`
- `policy_id`: `"gp_ucb"`
- `world_id`: `"rural_clinic"`
- 5 theory stages focused on uncertainty quantification, kernel approximation, beta parameter tuning.

#### 6.2 — Lesson: Ensemble Decision Making

- `lesson_id`: `"lesson_bootstrapped_ensemble"`
- `policy_id`: `"bootstrapped_ensemble"`
- `world_id`: `"moviematch"`
- 5 stages on bootstrap diversity, head count vs. performance, ensemble agreement.

#### 6.3 — Lesson: Hybrid Contextual Models

- `lesson_id`: `"lesson_linucb_hybrid"`
- `policy_id`: `"linucb_hybrid"`
- `world_id`: `"newsfeed"`
- 5 stages on shared vs. arm-specific features, decomposition benefits, dimension tradeoffs.

#### 6.4 — Lesson: Tree-Based Bandits

- `lesson_id`: `"lesson_tree_ucb"`
- `policy_id`: `"tree_ucb"`
- `world_id`: `"rural_clinic"`
- 5 stages on context bucketing, discretization tradeoffs, tree vs. linear.

#### 6.5 — Lesson: Thompson Tree Sampling

- `lesson_id`: `"lesson_tree_ts"`
- `policy_id`: `"tree_ts"`
- `world_id`: `"moviematch"`
- 5 stages on posterior sampling in buckets, uncertainty-driven exploration.

#### 6.6 — Lesson: Continuous Action Selection

- `lesson_id`: `"lesson_cats"`
- `policy_id`: `"cats"`
- `world_id`: `"rural_clinic"` (adapted for continuous action)
- 5 stages on action space, exploration radius, utility landscape.

#### 6.7 — Lesson: World-Specific Application

- Pair `ucb1` with a new world (ShopSmart or RidePilot).
- Demonstrates algorithm behavior in a fresh domain.

### Exit Criteria
- `LESSON_REGISTRY` has 14 lessons (8 existing + 6 new).
- Each new lesson has 5 theory stages, objective thresholds, and stage-locked controls.
- All lessons evaluate deterministically under fixed seeds.
- Lesson progression tests pass for all 14.

---

## 9) Phase 7: Simulation Loop Unification (3 items)

**Priority:** Medium — two parallel simulation systems should share a common abstraction.

### Objectives
Unify DiscreteSimulator and ContinuousSimulator behind a shared protocol, implement a ConfigurableContinuousWorld, and add regret tracking to continuous simulation.

### Deliverables

#### 9.1 — Shared simulator protocol

- File: `src/web/contracts.py` (amend)
- `Simulator` Protocol: `reset()`, `step() -> StepResult`, `run_steps(n)`, `replay_payload()`.
- `DiscreteSimulator` and `ContinuousSimulator` both implement `Simulator`.
- `SimulationStepResult` and `ContinuousStepResult` share a `StepResult` base or union.

#### 9.2 — ConfigurableContinuousWorld implementation

- File: `src/web/continuous/configurable_world.py` (new)
- Implements `ContinuousWorld` Protocol, backed by `WorldConfig`.
- Quadratic reward: `reward = max(0, 1 - (action - optimal_action)² / scale)`.
- `optimal_action` derived from feature-weighted base rate.
- Shares `WorldConfig` schema — no new config format.

#### 9.3 — Add regret to ContinuousSimulator

- File: `src/web/continuous/simulator.py`
- Accept `optimal_reward_fn` in constructor (same pattern as DiscreteSimulator).
- Compute per-step regret: `optimal_reward - actual_reward`.
- Track cumulative regret in state.
- Add `replay_payload()`.

### Exit Criteria
- `isinstance(discrete_sim, Simulator)` and `isinstance(continuous_sim, Simulator)` both True.
- Continuous world can use existing WorldConfig from clinic/moviematch/newsfeed.
- Continuous simulator tracks cumulative regret.
- Both simulators pass protocol compliance tests.

---

## 10) Phase 8: Full Test Suite (4 items)

**Priority:** High — release gate cannot pass without integration and smoke tests.

### Objectives
Build integration test harness, UI smoke tests, Phase 9 backend tests, and debug pane correctness tests.

### Deliverables

#### 10.1 — Integration test suite

- File: `tests/flet_redesign/test_integration.py` (new)
- Full step loop per policy-world pair (14 policies × 3 worlds = 42 combinations minimum).
- Lesson progression: start lesson → run steps → verify `evaluate_lesson_objective()` → advance stage → repeat through stage 5 → verify `completed`.
- Debug snapshot correctness: run 10 steps → call `get_debug_snapshot()` → verify keys match expected schema per policy family.
- Checkpoint round-trip: simulate → save checkpoint → load → resume → verify identical state.
- Target: +30 tests.

#### 10.2 — UI smoke tests

- File: `tests/flet_redesign/test_ui_smoke.py` (new)
- `build_route_ui_model()` for Home, Lesson, Arena, Sandbox, Comparison routes.
- Verify each returns non-None model with expected fields populated.
- Verify `build_shell_stack()` produces correct view hierarchy.
- Verify `normalize_route()` resolves edge cases.
- Verify `ParamControlSpec` defaults produce valid controls for all supported policies.
- Target: +15 tests.

#### 10.3 — Phase 9 backend tests

- Files: `tests/flet_redesign/test_comparison.py`, `tests/flet_redesign/test_sandbox.py` (new)
- `test_comparison.py`: orchestrator deterministic equality (same seed → same results), batch stats accuracy against known fixture, snapshot diff correctness.
- `test_sandbox.py`: editor validation (rejects invalid horizon, rates > 1, negative rates), `build_world_override()` produces correct overrides, scenario immutability.
- Target: +12 tests.

#### 10.4 — Debug pane correctness tests

- File: `tests/flet_redesign/test_debug_panes.py` (new)
- For each of 7 debug pane builders: supply known snapshot → verify output fields match.
- Test edge cases: empty snapshots, single arm, all arms with zero pulls.
- Target: +10 tests.

### Exit Criteria
- Integration tests: ≥30 passing, covering all policy-world pairs and lesson flows.
- UI smoke tests: ≥15 passing, covering all routes and model builders.
- Phase 9 tests: ≥12 passing for comparison and sandbox.
- Debug pane tests: ≥10 passing.
- Total test count: ≥159 (92 existing + 67 new).

---

## 11) Phase 9: Presentation Polish (12 items)

**Priority:** Medium — charts, tables, navigation, and presets make the app usable.

### Objectives
Add actual chart rendering, interactive trace tables, Comparison route, preset management, context inspection depth, and wire all disconnected UI utilities.

### Deliverables

#### 9.1 — Add chart rendering

- File: `src/web/ui/charts.py` (new)
- Use `flet` chart controls (`ft.LineChart`, `ft.BarChart`) or matplotlib-backed images.
- Cumulative reward chart (x: step, y: cumulative_reward).
- Cumulative regret chart (x: step, y: cumulative_regret).
- Arm pull distribution (bar chart per arm).
- Uncertainty view (for contextual policies: confidence bonus over time).
- Wire into `main.py` for arena and comparison routes.

#### 9.2 — Add trace table rendering

- File: `src/web/ui/components/trace_table.py` (new)
- Interactive `ft.DataTable` with columns: step, chosen_arm, reward, cumulative_reward, cumulative_regret.
- Filter/search by arm name.
- Export buttons (JSON, CSV) using `TraceBuffer.to_json()`/`to_csv()`.
- Virtual scrolling for large traces (>1000 rows).

#### 9.3 — Add Comparison route

- File: `src/web/router.py`
- Add `COMPARISON = "/comparison"` to `AppRoute` enum.
- Add `RouteSpec` with title "Comparison", heading "Comparison Workspace".
- Add to `_ROUTE_SPECS`.
- Add to `normalize_route()` matching.
- Wire to comparison page in `main.py`.

#### 9.4 — Wire PresetManager into UI

- Files: `src/web/main.py`, `src/web/preset_manager.py`
- Add "Save Preset" button to arena and sandbox pages.
- Add "Load Preset" dropdown.
- Persist to `~/.coba_flet_presets.json`.

#### 9.5 — Wire checkpoint save/load into UI

- File: `src/web/main.py`
- Add "Save Checkpoint" button during simulation runs.
- Add "Load Checkpoint" on arena route entry (if checkpoint file exists).
- Verify checkpoint integrity before loading.

#### 9.6 — Wire SandboxEditor build_world_override()

- File: `src/web/ui/pages/sandbox_page.py`
- Arm base rate sliders call `SandboxEditor.build_world_override()`.
- Resulting `WorldConfig` is displayed in a read-only preview panel.
- "Run with Overrides" button creates a `ConfigurableWorld` from the override and runs.

#### 9.7 — Wire AppStateStore

- File: `src/web/main.py`
- Replace ad-hoc world/policy tracking with `AppStateStore`.
- Use `AppStateStore.switch_world()` and `build_world()` for world changes.

#### 9.8 — Deepen context inspection

- File: `src/web/main.py`
- After each step, call `context_to_vector()` on live context.
- Render feature vector in right panel with per-feature values.
- For contextual policies, show which features contribute most to the chosen arm's score.

#### 9.9 — Wire render_theory_stage_markdown to Flet rendering

- File: `src/web/main.py`
- Lesson route renders `render_theory_stage_markdown()` output as `ft.Markdown()`.
- Stage cards update when stage advances.

#### 9.10 — Wire explain_step_delta with live data

- File: `src/web/main.py`
- After each step, call `explain_step_delta(prev_trace_record, current_trace_record)`.
- Display in lesson right panel.

#### 9.11 — Enforce locked_control_keys_for_stage in UI

- File: `src/web/main.py`
- On lesson route, read `locked_control_keys_for_stage(lesson, stage)`.
- Disable corresponding ParamControlSpec controls (set `disabled=True` on Flet controls).

#### 9.12 — Wire evaluate_lesson_objective mid-run

- File: `src/web/main.py`
- After each step in lesson mode, call `evaluate_lesson_objective()`.
- If true, show "Objective Complete" badge, enable stage advance button.
- If all 5 stages complete, show "Lesson Complete" and offer next lesson.

### Exit Criteria
- Charts render live reward/regret/arm distribution data.
- Trace table supports filter, search, and export.
- Comparison route navigable from nav bar.
- Presets and checkpoints save/load correctly.
- Sandbox overrides produce valid custom worlds.
- Lesson UI has full stage progression with locked controls and objective evaluation.

---

## 12) Phase 10: Docs and Release (3 items)

**Priority:** Medium — required for release gate.

### Objectives
Write contributor documentation, architecture overview, release checklist, and prepare RC tag.

### Deliverables

#### 10.1 — Contributor guides

- Files: `docs/contributing/worlds.md`, `docs/contributing/policies.md`, `docs/contributing/lessons.md`
- Adding a world: schema, ConfigurableWorld, core_worlds registration, presets, tests.
- Adding a policy: BanditPolicy protocol, DebugSnapshotProvider, policy_factory registration, policy_capabilities, debug pane, tests.
- Adding a lesson: LessonConfig, theory stages, objectives, stage_locked_controls, LESSON_REGISTRY, tests.

#### 10.2 — Architecture overview

- File: `docs/web_architecture.md`
- Layer diagram, module dependency graph, data flow from simulator → view model → Flet rendering.
- Protocol hierarchy, factory pattern, capability system, pedagogical staging.
- How debug panes, comparison orchestrator, and sandbox editor fit in.

#### 10.3 — Release checklist and RC tag

- File: `docs/release_checklist.md`
- All 10 phase gates checked.
- Quality gates: lint, types, unit tests (≥159), integration tests (≥30), smoke tests (≥15), deterministic replay, performance baseline.
- RC tag instructions: `git tag v0.1.0-rc1`, verify demo scenario on all worlds/lessons.

### Exit Criteria
- All 3 contributor guides exist and are verified by a reviewer.
- Architecture doc is accurate against final codebase.
- Release checklist has zero unchecked items.
- RC tag created and demo passes.

---

## 13) Phase Dependency Graph

```
Phase 1 (Interactive Core) ─────────────────────────────────────────┐
  │                                                                  │
  ├── Phase 2 (UI Pages) ─── requires Phase 1 (sim running)          │
  │     │                                                            │
  │     └── Phase 9 (Polish) ─── requires Phase 2 (pages exist)     │
  │                                                                  │
  ├── Phase 3 (Algorithms) ─── independent                          │
  │     │                                                            │
  │     └── Phase 6 (Lessons) ─── requires Phase 3 (LinTS)          │
  │                                                                  │
  ├── Phase 4 (Debugger) ─── independent                            │
  │                                                                  │
  ├── Phase 5 (Worlds) ─── independent                              │
  │                                                                  │
  ├── Phase 7 (Unification) ─── independent                         │
  │                                                                  │
  ├── Phase 8 (Tests) ─── requires Phases 1-7                       │
  │                                                                  │
  └── Phase 10 (Docs) ─── requires Phases 1-9                       │
```

Phases 1, 3, 4, 5, and 7 are parallelizable. Phase 2 depends on Phase 1. Phases 6, 8, 9, and 10 are sequential.

---

## 14) Item-to-Phase Mapping

| ID# | Item | Phase | File(s) |
|---|---|---|---|
| 1 | Wire simulator to Play/Pause/Step | 1 | `main.py` |
| 2 | Wire live state to UI | 1 | `main.py` |
| 3 | Wire RunController to simulator | 1 | `main.py` |
| 4 | Connect PreferencesStore | 1 | `main.py`, `preferences.py` |
| 5 | Connect lesson stage advancement | 1 | `main.py`, `lessons.py` |
| 6 | Wire ParamControlSpec to live updates | 1 | `main.py`, `param_controls.py` |
| 7 | Create ui/pages/ directory | 2 | `ui/pages/__init__.py` |
| 8 | Build comparison page | 2 | `ui/pages/comparison_page.py` |
| 9 | Build sandbox page | 2 | `ui/pages/sandbox_page.py` |
| 10 | Build snapshot diff view | 2 | `ui/components/snapshot_diff_view.py` |
| 11 | Build batch summary panel | 2 | `ui/components/batch_summary_panel.py` |
| 12 | Implement LinTS | 3 | `policies/lints_policy.py` |
| 13 | Implement Drift Detector A | 3 | `drift/cusum_detector.py` |
| 14 | Implement Drift Detector B | 3 | `drift/adwin_detector.py` |
| 15 | Context-free debug snapshots | 4 | 5 `policies/*.py` files |
| 34 | Deepen debug panes | 4 | `debug/contextual.py`, `advanced.py`, `continuous.py` |
| 35 | Call debug snapshots during sim | 4 | `main.py`, `simulator.py` |
| 16 | ShopSmart world | 5 | `worlds/core_worlds.py` |
| 17 | RidePilot world | 5 | `worlds/core_worlds.py` |
| 18 | GameBot world | 5 | `worlds/core_worlds.py` |
| 19 | LabTrial world | 5 | `worlds/core_worlds.py` |
| 20 | gp_ucb lesson | 6 | `curriculum/lessons.py` |
| 21 | bootstrapped_ensemble lesson | 6 | `curriculum/lessons.py` |
| 22 | linucb_hybrid lesson | 6 | `curriculum/lessons.py` |
| 23 | tree_ucb lesson | 6 | `curriculum/lessons.py` |
| 24 | tree_ts lesson | 6 | `curriculum/lessons.py` |
| 25 | cats lesson | 6 | `curriculum/lessons.py` |
| 26 | World-specific application lesson | 6 | `curriculum/lessons.py` |
| 27 | Integration test suite | 8 | `tests/.../test_integration.py` |
| 28 | UI smoke tests | 8 | `tests/.../test_ui_smoke.py` |
| 29 | Phase 9 backend tests | 8 | `tests/.../test_comparison.py`, `test_sandbox.py` |
| 30 | Debug pane correctness tests | 8 | `tests/.../test_debug_panes.py` |
| 31 | Shared simulator protocol | 7 | `contracts.py` |
| 32 | ConfigurableContinuousWorld | 7 | `continuous/configurable_world.py` |
| 33 | Regret in ContinuousSimulator | 7 | `continuous/simulator.py` |
| 36 | Add Comparison route | 9 | `router.py` |
| 37 | Wire PresetManager | 9 | `main.py`, `preset_manager.py` |
| 38 | Add chart rendering | 9 | `ui/charts.py` |
| 39 | Add trace table rendering | 9 | `ui/components/trace_table.py` |
| 40 | Deepen context inspection | 9 | `main.py` |
| 41 | Render theory_stage_markdown | 9 | `main.py` |
| 42 | Wire explain_step_delta | 9 | `main.py` |
| 43 | Wire evaluate_lesson_objective | 9 | `main.py` |
| 44 | Enforce locked controls in UI | 9 | `main.py` |
| 45 | Wire AppStateStore | 9 | `main.py`, `state_store.py` |
| 46 | Wire build_world_override | 9 | `ui/pages/sandbox_page.py` |
| 47 | Wire checkpoint save/load | 9 | `main.py` |
| 48 | Contributor docs | 10 | `docs/contributing/*.md` |
| 49 | Architecture overview | 10 | `docs/web_architecture.md` |
| 50 | Release checklist + RC tag | 10 | `docs/release_checklist.md` |

All 50 items accounted for.

---

## 15) Quality Gates Per Phase

| Phase | Lint | Types | Unit Tests | Integration | Smoke | Replay |
|---|---|---|---|---|---|---|
| 1 | ☑ | ☑ | 92+ | — | — | ☑ |
| 2 | ☑ | ☑ | 92+ | — | — | ☑ |
| 3 | ☑ | ☑ | 98+ | — | — | ☑ |
| 4 | ☑ | ☑ | 104+ | — | — | ☑ |
| 5 | ☑ | ☑ | 110+ | — | — | ☑ |
| 6 | ☑ | ☑ | 116+ | — | — | ☑ |
| 7 | ☑ | ☑ | 120+ | — | — | ☑ |
| 8 | ☑ | ☑ | 159+ | ☑ | ☑ | ☑ |
| 9 | ☑ | ☑ | 159+ | ☑ | ☑ | ☑ |
| 10 | ☑ | ☑ | 159+ | ☑ | ☑ | ☑ |

---

## 16) Immediate Next Actions (Phase 1)

1. Add `_create_simulator()` factory to `main.py` — instantiate `DiscreteSimulator` with current selections.
2. Attach `on_click` to Play/Pause/Step/Reset buttons — call `RunController` methods and `simulator.step()`.
3. After each step, call `build_route_ui_model()` with live trace data and `page.update()`.
4. Wire `PreferencesStore` — save on change, load on route entry.
5. Verify: select world+policy, press Play, see live step counter and reward update.
