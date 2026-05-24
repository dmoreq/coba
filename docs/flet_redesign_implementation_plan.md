# COBA Flet Redesign Implementation Plan

Last updated: 2026-05-24
Source baseline: `docs/flet_redesign_plan.md`

---

## 1) Purpose and Scope

This document converts the redesign architecture into an executable implementation plan with concrete deliverables, dependencies, acceptance criteria, and release gates.

In scope:
- Flet application redesign architecture and component system.
- Seven narrative worlds and shared world schema.
- Full curriculum path (17 lessons + sandbox).
- Algorithm debugger framework and algorithm-specific debug renderers.
- Configuration controls with tooltip guidance.
- Testing, observability, and release readiness.

Out of scope (for this plan version):
- New external identity/auth systems.
- Multi-user collaboration backend.
- Native mobile app targets.

---

## 2) Delivery Principles

1. Discrete-first implementation, contextual/advanced features layered after stable core.
2. Every phase ships as a usable increment (no phase ends in design-only artifacts).
3. Determinism first: seeded runs, reproducible traces, and regression tests before feature expansion.
4. Debugger parity: no algorithm marked complete without debug snapshot support.
5. Curriculum quality gate: lesson logic, theory content, and interaction controls must ship together.

---

## 3) Workstreams and Ownership Model

### A. Engine
- Bandit policy interfaces and implementations.
- Simulation loop, world adapters, reward generation.
- Snapshot/debug schema and trace capture.

### B. UI/UX
- Page shell and navigation.
- Scene/treatment panels, controls, charts, debugger layout.
- Tooltip and hover help system.

### C. Curriculum
- Lesson metadata and progression.
- Theory card content and pedagogical sequencing.
- Scenario-specific tasks and validation criteria.

### D. QA/Infra
- Unit/integration tests, deterministic fixtures, CI checks.
- Profiling and performance baselines.
- Packaging, docs quality gates.

---

## 4) Proposed Project Structure (Implementation Target)

```text
src/coba/
  app/
    main.py
    router.py
    state_store.py
  engine/
    interfaces.py
    simulator.py
    trace.py
    debug/
      schema.py
      collector.py
      render_registry.py
    policies/
      context_free/
      linear/
      bayesian/
      ensemble/
      continuous/
  worlds/
    schema.py
    base.py
    clinic.py
    moviematch.py
    newsfeed.py
    shopsmart.py
    ridepilot.py
    gamebot.py
    labtrial.py
  curriculum/
    lessons/
    theory/
    progression.py
  ui/
    pages/
    components/
      scene_panel.py
      treatment_card.py
      param_controls.py
      tooltips.py
      debugger/
    charts/
  tests/
    unit/
    integration/
    ui_smoke/
```

Notes:
- Keep to current repo conventions where they differ; this is the logical target split.
- Merge modules only when complexity stays low and interfaces remain explicit.

---

## 5) Milestone Plan (55-Day Roadmap)

## Phase 0 (Day 1): Scaffold + Contracts

### Objectives
- Establish skeleton architecture and technical guardrails.

### Deliverables
- App bootstrap and route shell (`Home`, `Lesson`, `Arena`, `Sandbox`).
- Core interfaces: `BanditPolicy`, `World`, `SimulationStepResult`, `DebugSnapshotProvider`.
- Tooling: lint, typing, tests, pre-commit, CI baseline.

### Key Tasks
1. Create package structure and minimal import boundaries.
2. Define config loading and environment profile.
3. Add smoke test: app starts and one dummy simulation step executes.

### Exit Criteria
- CI passes lint + tests on clean checkout.
- App launches locally with no runtime errors.

---

## Phase 1 (Days 2-4): Discrete Engine + State Core

### Objectives
- Build deterministic simulation core for discrete-action algorithms.

### Deliverables
- State models: `SimulationState`, `ArmState`, `RunConfig`, `TraceBuffer`.
- Policies: Random, Epsilon-Greedy, UCB1, Thompson (Beta-Bernoulli), Softmax.
- Seeded replay mode and trace export (`json/csv`).

### Key Tasks
1. Implement engine step contract (`context -> select -> reward -> update -> snapshot`).
2. Add reward generators (Bernoulli, Gaussian) with fixed-seed fixtures.
3. Add regret and cumulative reward computations.

### Exit Criteria
- Deterministic equality across repeated seeds.
- UCB1/TS outperform random baseline in regression tests.

---

## Phase 2 (Days 5-6): Core Worlds (First Three)

### Objectives
- Integrate narrative world layer with schema validation.

### Deliverables
- World schema (`WorldConfig`, `FeatureDef`, `ArmDef`) and parser/validator.
- Worlds: Rural Clinic, MovieMatch, NewsFeed.
- Difficulty presets and world metadata for UI rendering.

### Key Tasks
1. Implement per-world reward and context generators.
2. Add world registry and selection API.
3. Add fixtures for deterministic world behavior.

### Exit Criteria
- World switching works without restart.
- Each world has at least two validated presets.

---

## Phase 3 (Days 7-9): UI Shell + Control System

### Objectives
- Build core interaction surface and parameter control system.

### Deliverables
- Three-pane layout (scene/action/metrics-debug).
- `ScenePanel`, `TreatmentCard`, run controls (`step/play/pause/reset`).
- Param control renderer using `ParamControlSpec`.
- Tooltip engine with formula, intuition, and tuning guidance.

### Key Tasks
1. Implement UI state sync with simulation state store.
2. Add guarded live parameter updates.
3. Persist local UI preferences (speed, last world, algorithm).

### Exit Criteria
- Users can run, pause, reset, and adjust controls safely.
- No major layout overflow on desktop target resolution.

---

## Phase 4 (Days 10-13): Arena + Charts + Trace

### Objectives
- Provide real-time observability of algorithm behavior.

### Deliverables
- Arena execution page.
- Charts: cumulative reward, regret, arm pulls, uncertainty view.
- Step trace table with filter/search and export.
- Run snapshot persistence for before/after comparison.

### Key Tasks
1. Decouple chart refresh cadence from engine step cadence.
2. Add trace serialization and replay loader.
3. Validate chart values against trace source-of-truth.

### Exit Criteria
- Live charts are stable and consistent with trace entries.
- Export/import replay produces matching metrics.

---

## Phase 5 (Days 14-18): Foundation Lessons

### Objectives
- Ship first curriculum segment with pedagogy + mechanics integrated.

### Deliverables
- Lesson engine for first five lessons.
- 5-stage theory card renderer and lesson objective logic.
- Guided interactions with locked/unlocked controls by stage.

### Key Tasks
1. Define lesson metadata schema and progression state.
2. Add objective validators (regret/reward/stability thresholds).
3. Add “explain this step” from debug deltas.

### Exit Criteria
- Full completion flow works for first learning path.
- Lesson outcome evaluation is deterministic under fixed seeds.

---

## Phase 6 (Days 19-25): Contextual Lessons

### Objectives
- Introduce contextual decisioning and feature-aware pedagogy.

### Deliverables
- LinUCB, LinUCB-SW, Logistic bandit implementations.
- Feature influence panel and context inspection UI.
- Contextual lesson branch with world presets.

### Key Tasks
1. Implement matrix/vector update paths with numerical safeguards.
2. Add algorithm-specific debug snapshots for linear/logistic updates.
3. Add regression tests for update formulas and confidence bounds.

### Exit Criteria
- Contextual policies converge on synthetic test fixtures.
- Debug panes surface all required internals per lesson.

---

## Phase 7 (Days 26-36): Advanced Discrete + Ensembles

### Objectives
- Expand algorithm inventory and preserve UI/debugger coherence.

### Deliverables
- GP-UCB, Bootstrapped Ensemble, Tree UCB/TS, LinUCB Hybrid.
- Capability flag system to drive debugger panel composition.
- Comparative diagnostics panel (adaptation lag, uncertainty quality).

### Key Tasks
1. Add algorithm registry metadata (`group`, `needs_context`, `debug_views`).
2. Implement test suite per algorithm against baseline expectations.
3. Tune performance hotspots for high-step simulations.

### Exit Criteria
- All added algorithms support trace + debugger integration.
- UI does not branch into algorithm-specific hardcoded layouts.

---

## Phase 8 (Days 37-45): Continuous + Production Features

### Objectives
- Deliver continuous-action support and operational reliability features.

### Deliverables
- CATS implementation and continuous-action debugger panes.
- Drift detection integration and event visualization.
- Checkpoint/save/resume flows and preset management.

### Key Tasks
1. Add continuous action control widgets and validation constraints.
2. Implement drift detector with configurable sensitivity.
3. Add checkpoint metadata integrity checks.

### Exit Criteria
- Continuous algorithms run stably and are explainable via debugger.
- Checkpoint restore yields equivalent resumed behavior.

---

## Phase 9 (Days 46-48): Comparison + Sandbox Backend

### Objectives
- Build multi-policy comparison orchestrator and sandbox editor engine.

### Deliverables
- Multi-policy comparison runner under identical seeds/world streams.
- Batch repeated-seed execution with summary stats (mean/std/CI95).
- Snapshot diff tools for trace and debugger states.
- Sandbox editor with config validation and world overrides.

### Key Tasks
1. Build synchronized run orchestrator across N policies (`comparison/orchestrator.py`).
2. Add repeated-seed batch execution with stats aggregation (`comparison/stats.py`).
3. Build snapshot diff helpers for trace/debugger (`comparison/snapshot_diff.py`).
4. Implement sandbox editor with validation guards (`sandbox.py`).

### Exit Criteria
- Orchestrator produces reproducible deterministic output.
- Snapshot diff correctly identifies changed keys.
- Sandbox editor rejects invalid configurations.
- All Phase 9 backend modules have unit tests.

**Status: Backend modules complete (70%). No UI integration yet.**

---

## Phase 9.5 (Days 49-50): Comparison + Sandbox UI + Tests

### Objectives
- Wire Phase 9 backend into Flet shell with interactive pages.

### Deliverables
- `ui/pages/comparison_page.py` — side-by-side policy comparison page.
- `ui/pages/sandbox_page.py` — sandbox editor page.
- `ui/components/snapshot_diff_view.py` — diff renderer component.
- `ui/components/batch_summary_panel.py` — stats summary panel.
- Full test suite for comparison and sandbox modules.
- End-to-end flow: select → run → view results → export.

### Key Tasks
1. Create `ui/pages/` directory structure.
2. Build comparison page: policy picker, run controls, charts, results table, diff viewer.
3. Build sandbox page: world/param editor, validate, run, inspect results.
4. Add route view-model builders for comparison/sandbox in `view_models.py`.
5. Add rendering branches in `main.py` for comparison/sandbox views.
6. Write complete test suite (unit + integration for orchestrator, stats, diff, sandbox).

### Exit Criteria
- Side-by-side comparisons are reproducible and exportable.
- Sandbox can create and run custom scenarios without crashes.
- All comparison/sandbox tests pass (target: ~105 total tests).

---

## Phase 10 (Days 51-55): Polish + Release + Gap Closure

### Objectives
- Stabilize product quality, performance, and onboarding docs. Close remaining algorithm/world/debugger/test gaps.

### Deliverables

#### UX Polish
- Spacing, typographic hierarchy, interaction transitions, keyboard navigation, dark mode.

#### Performance
- Steps/sec benchmark per algorithm class, UI update latency profiling, memory baseline for 10k+ step runs.

#### Gap Closure
- Context-free policy debuggers (Random, Epsilon-Greedy, UCB1, Thompson, Softmax — all missing).
- LinTS policy implementation (Linear Thompson Sampling with posterior update).
- Drift Detector A and B implementations.
- Remaining worlds: ShopSmart, RidePilot, GameBot, LabTrial.
- Integration test suite (full step loop per policy-world pair, lesson progression, debug snapshot correctness).
- UI smoke test suite (route load, run controls, world switch, lesson completion, chart refresh, trace table).

#### Documentation
- Contributor guides: adding a world, policy, lesson.
- Architecture overview, release checklist, RC tag.

### Key Tasks
1. Fix high-severity UI and logic defects.
2. Execute full regression suite and performance baseline.
3. Implement missing algorithms (LinTS, Drift A/B) with debugger + tests.
4. Implement missing worlds (ShopSmart, RidePilot, GameBot, LabTrial) with presets.
5. Add context-free policy debugger renderers (arm-level counts/reward/posterior snapshots).
6. Build integration and UI smoke test suites.
7. Publish final architecture and extension documentation.

### Exit Criteria
- All release gates pass (lint, types, unit tests, integration tests, UI smoke tests, deterministic replay, performance baseline).
- Target: 17 algorithms implemented (all with debugger + tests).
- Target: 7 worlds integrated with presets.
- Target: 8 configured lessons executable end-to-end.
- RC demo scenario succeeds on all target worlds/lessons.
- No blocker/high defects open.

---

## 6) Algorithm Rollout Matrix (17 Variants)

| Group | Algorithms | Phase | Status | Notes |
|---|---|---|---:|---|---|
| Context-Free | Random, Epsilon-Greedy, UCB1, Thompson, Softmax | 1, 5 | Impl:☑ Debug:☐ | All 5 implemented; none have debugger renderers |
| Linear Contextual | LinUCB, LinUCB-SW, LinTS | 6 | LinUCB:☑ SW:☑ TS:☐ | LinTS not implemented |
| Logistic | Logistic UCB/TS variant | 6 | ☑ | Implemented with debugger + tests |
| Bayesian/Kernel | GP-UCB | 7 | ☑ | Implemented with debugger + tests |
| Ensemble | Bootstrapped Ensemble | 7 | ☑ | Implemented with debugger + tests |
| Hybrid | LinUCB Hybrid | 7 | ☑ | Implemented with debugger + tests |
| Tree | RF-UCB, RF-TS | 7 | ☑ | Both implemented with debugger + tests |
| Continuous | CATS | 8 | ☑ | Implemented with debugger + tests |
| Drift-Aware | Drift detector variants (A/B) | 8 | ☐ | Neither implemented — planned for Phase 10 |

---

## 7) Debugger Architecture Delivery Plan

1. Core schema and collector (`AlgorithmDebugSnapshot`, `PerArmDebugState`, `ContinuousDebugState`) in Phase 1.
2. Renderer registry by algorithm group in Phase 4.
3. Group-specific panes in Phases 6-8.
4. Cross-policy snapshot diff in Phase 9.

Hard rule:
- A policy is not considered complete until its debug snapshot + renderer + tests are included.

---

## 8) Testing Strategy and Quality Gates

### Unit Tests (92 passing, 0 for Phase 9)
- Policy updates and selection decisions.
- Reward generators and world schema validation.
- Serialization contracts for traces and snapshots.

### Integration Tests (Not Started)
- Full step loop per policy-world pair.
- Lesson progression and objective checks.
- Debug snapshot correctness against known fixtures.

### UI Smoke Tests (Not Started)
- Route load, run controls, world switch, lesson completion.
- Chart refresh and trace table interactions.

### Performance Benchmarks (Passing)
- Steps/sec threshold by algorithm class.
- UI update latency budget under streaming traces.

### Release Gate (must pass)
1. Lint + type checks. ☑
2. Unit tests (92 passing). ☑
3. Integration tests. ☐
4. UI smoke tests. ☐
5. Deterministic replay checks. ☑
6. Performance baseline. ☑
7. No blocker/high defects open. ☐

---

## 9) Risks and Mitigations

| Risk | Impact | Mitigation | Status |
|---|---|---|---|
| Debugger complexity explosion | High | Capability flags + renderer registry, no ad-hoc branches | Mitigated |
| Numerical instability in contextual models | High | Regularization defaults, NaN guards, formula tests | Mitigated |
| Feature scope creep across 17 variants | High | Phase gates + strict definition of "done" | Active: 13/17 implemented |
| UI performance regressions | Medium | Batched updates, throttled chart rendering | Monitoring |
| Content bottleneck for lessons/tooltips | Medium | Template-driven authoring and parallel content stream | Monitoring |
| Phase 9 backend disconnected from UI | High | Split into Phase 9 (backend) + Phase 9.5 (UI + tests) | New: backend done, UI not started |
| No integration or UI smoke test suites | High | Add to Phase 9.5 + Phase 10 scope | New: zero tests exist |
| Context-free policy debugger gap | Medium | 5 policies lack debug renders — add in Phase 10 | New: violates debugger parity |
| World coverage gap (3/7) | Medium | 4 worlds need implementation — add to Phase 10 | New: 57% coverage |
| Release blocked on E2E validation | Critical | Gate Phase 10 exit on full test suite green | New: no release without integration + smoke tests |

---

## 10) Definition of Done (Per Feature)

A feature is done only when all are true:
1. Implementation merged and integrated.
2. Tests added/updated and passing.
3. Trace/debug support included where applicable.
4. User-facing controls and tooltips documented.
5. Added to tracker with verification evidence.

---

## 11) Weekly Execution Cadence

1. Monday: plan lock + dependency review + risk review.
2. Daily: standup updates in tracker (status, blockers, next actions).
3. Mid-week: integration checkpoint and deterministic replay spot-check.
4. Friday: phase gate review + demo + tracker evidence update.

---

## 12) Immediate Next Actions (Phase 9.5)

1. Create `src/web/ui/pages/` directory structure.
2. Build `comparison_page.py` — policy picker → run orchestrator → results (charts, table, diff viewer).
3. Build `sandbox_page.py` — world/param editor → validate → run → inspect.
4. Add route view-model builders in `view_models.py` for comparison/sandbox.
5. Add rendering branches in `main.py` for comparison/sandbox views.
6. Write Phase 9 + 9.5 test suite (orchestrator, stats, diff, sandbox editor, page models).
7. Add Comparison route if needed to `router.py` (currently only `/sandbox`).
