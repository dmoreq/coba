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

## Phase 9 (Days 46-50): Comparison + Sandbox

### Objectives
- Enable robust experimentation and side-by-side policy evaluation.

### Deliverables
- Multi-policy comparison mode under identical seeds/world streams.
- Open sandbox editor for world and algorithm parameters.
- Snapshot diff tools for trace and debugger states.

### Key Tasks
1. Build synchronized run orchestrator across N policies.
2. Add repeated-seed batch execution summary (mean/variance/confidence).
3. Add sandbox safety guards to prevent invalid configs.

### Exit Criteria
- Side-by-side comparisons are reproducible and exportable.
- Sandbox can create and run custom scenarios without crashes.

---

## Phase 10 (Days 51-55): Polish + Docs + Release

### Objectives
- Stabilize product quality, performance, and onboarding docs.

### Deliverables
- UX polish pass (spacing, typographic hierarchy, interaction transitions).
- Performance pass and benchmark report.
- Contributor docs for adding a world/policy/lesson.
- Release checklist and RC tag.

### Key Tasks
1. Fix high-severity UI and logic defects.
2. Execute full regression suite and performance baseline.
3. Publish final architecture and extension documentation.

### Exit Criteria
- All release gates pass.
- RC demo scenario succeeds on all target worlds/lessons.

---

## 6) Algorithm Rollout Matrix (17 Variants)

| Group | Algorithms | Phase | Debugger Requirement |
|---|---|---:|---|
| Context-Free | Random, Epsilon-Greedy, UCB1, Thompson, Softmax | 1, 5 | Arm-level counts/reward/posterior |
| Linear Contextual | LinUCB, LinUCB-SW, LinTS | 6 | A/b matrices, theta, bonus terms |
| Logistic | Logistic UCB/TS variant | 6 | Gradient, Hessian/approx, confidence proxy |
| Bayesian/Kernel | GP-UCB | 7 | Posterior mean/variance, kernel state |
| Ensemble | Bootstrapped Ensemble | 7 | Per-model prediction and aggregated uncertainty |
| Hybrid | LinUCB Hybrid | 7 | Shared + arm-specific decomposition |
| Tree | RF-UCB, RF-TS | 7 | Tree vote stats + confidence proxy |
| Continuous | CATS | 8 | Action sample distribution and utility landscape |
| Drift-Aware | Drift detector variants | 8 | Detector state and trigger rationale |

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

### Unit Tests
- Policy updates and selection decisions.
- Reward generators and world schema validation.
- Serialization contracts for traces and snapshots.

### Integration Tests
- Full step loop per policy-world pair.
- Lesson progression and objective checks.
- Debug snapshot correctness against known fixtures.

### UI Smoke Tests
- Route load, run controls, world switch, lesson completion.
- Chart refresh and trace table interactions.

### Performance Benchmarks
- Steps/sec threshold by algorithm class.
- UI update latency budget under streaming traces.

### Release Gate (must pass)
1. Lint + type checks.
2. Unit + integration + smoke tests.
3. Deterministic replay checks.
4. No blocker/high defects open.

---

## 9) Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Debugger complexity explosion | High | Capability flags + renderer registry, no ad-hoc branches |
| Numerical instability in contextual models | High | Regularization defaults, NaN guards, formula tests |
| Feature scope creep across 17 variants | High | Phase gates + strict definition of “done” |
| UI performance regressions | Medium | Batched updates, throttled chart rendering |
| Content bottleneck for lessons/tooltips | Medium | Template-driven authoring and parallel content stream |

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

## 12) Immediate Next Actions (Week 1)

1. Create/confirm module boundaries for `engine`, `worlds`, `ui`, `curriculum`.
2. Implement policy interfaces and deterministic simulation loop.
3. Deliver first three discrete policies with fixtures.
4. Add CI baseline and smoke launch test.
5. Start world schema implementation with one world (Clinic) as reference.
