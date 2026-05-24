# COBA Flet Redesign Progress Tracker

Last updated: 2026-05-24
Plan reference: `docs/flet_redesign_implementation_plan.md`
Architecture reference: `docs/flet_redesign_plan.md`

---

## 1) Status Legend

- `NS` = Not Started
- `IP` = In Progress
- `BL` = Blocked
- `RV` = In Review
- `DN` = Done

---

## 2) Program Snapshot

| Field | Value |
|---|---|
| Program Owner | _TBD_ |
| Tech Lead | _TBD_ |
| Start Date | _TBD_ |
| Target End Date | _TBD_ |
| Current Phase | Phase 9 |
| Overall Status | IP |
| Overall Completion | 80% |
| Top Blocker | Phase 9 has no UI pages or tests — backend modules exist but are disconnected from Flet shell |

---

## 3) Milestone Tracker

| Phase | Window | Status | Completion | Owner | Gate |
|---|---|---|---:|---|---|
| Phase 0: Scaffold + Contracts | Day 1 | DN | 100% | _TBD_ | ☑ |
| Phase 1: Discrete Engine + State | Days 2-4 | DN | 100% | _TBD_ | ☑ |
| Phase 2: Core Worlds | Days 5-6 | DN | 100% | _TBD_ | ☑ |
| Phase 3: UI Shell + Controls | Days 7-9 | DN | 100% | _TBD_ | ☑ |
| Phase 4: Arena + Charts + Trace | Days 10-13 | DN | 100% | _TBD_ | ☑ |
| Phase 5: Foundation Lessons | Days 14-18 | DN | 100% | _TBD_ | ☑ |
| Phase 6: Contextual Lessons | Days 19-25 | DN | 100% | _TBD_ | ☑ |
| Phase 7: Advanced Discrete + Ensembles | Days 26-36 | DN | 100% | _TBD_ | ☑ |
| Phase 8: Continuous + Production Features | Days 37-45 | DN | 100% | _TBD_ | ☑ |
| Phase 9: Comparison + Sandbox (Backend) | Days 46-48 | IP | 70% | _TBD_ | ☐ |
| Phase 9.5: Comparison + Sandbox (UI + Tests) | Days 49-50 | NS | 0% | _TBD_ | ☐ |
| Phase 10: Polish + Release | Days 51-55 | NS | 0% | _TBD_ | ☐ |

---

## 4) Phase Checklists

## Phase 0: Scaffold + Contracts

Status: `DN`
Owner: _TBD_
Target Date: _TBD_

- [x] Package structure scaffolded
- [x] App shell routes (`Home`, `Lesson`, `Arena`, `Sandbox`)
- [x] Core interfaces defined (`BanditPolicy`, `World`, `SimulationStepResult`)
- [x] Tooling configured (`ruff`, `mypy`, `pytest`, pre-commit)
- [x] CI baseline pipeline passing
- [x] Smoke test for app startup

Evidence:
- PR(s): local commit(s) on `main` (Phase 0 Steps 1-4)
- Test run: `pytest tests/flet_redesign -q` (16 passed)
- Quality run: `uv run --with pre-commit pre-commit run --all-files` (all hooks passed)
- CI workflow: `.github/workflows/flet-redesign-ci.yml`

---

## Phase 1: Discrete Engine + State

Status: `DN`
Owner: _TBD_
Target Date: _TBD_

- [x] `SimulationState`, `ArmState`, `RunConfig` implemented
- [x] Trace buffer and serialization implemented
- [x] Random policy integrated
- [x] Epsilon-Greedy policy integrated
- [x] UCB1 policy integrated
- [x] Thompson Sampling (Beta-Bernoulli) integrated
- [x] Softmax policy integrated
- [x] Deterministic seed replay verified
- [x] Baseline regret tests passing

Evidence:
- PR(s): local commit(s) on `main` (Phase 1 Steps 1-2)
- Test run: `pytest tests/flet_redesign -q -p no:asyncio` (40 passed)
- Quality run: `uv run --with pre-commit pre-commit run --all-files` (all hooks passed)

---

## Phase 2: Core Worlds (Clinic, MovieMatch, NewsFeed)

Status: `DN`
Owner: _TBD_
Target Date: _TBD_

- [x] World schema (`WorldConfig`, `FeatureDef`, `ArmDef`) implemented
- [x] Validator + parser implemented
- [x] Rural Clinic world integrated
- [x] MovieMatch world integrated
- [x] NewsFeed world integrated
- [x] Difficulty presets per world
- [x] World switching UI/API connected
- [x] World fixtures committed

Evidence:
- PR(s): local commit(s) on `main` (Phase 2 Step 1)
- Test run: `pytest tests/flet_redesign -q -p no:asyncio` (51 passed)
- World fixtures: `tests/flet_redesign/fixtures/core_world_fixtures.json`

---

## Phase 3: UI Shell + Controls

Status: `DN`
Owner: _TBD_
Target Date: _TBD_

- [x] Three-pane layout implemented
- [x] `ScenePanel` component implemented
- [x] `TreatmentCard` component implemented
- [x] Run controls implemented (`step/play/pause/reset`)
- [x] Param controls rendered from `ParamControlSpec`
- [x] Tooltip component implemented with formula/intuition guidance
- [x] UI state persistence (local preferences)
- [x] Desktop layout sanity check passed

Evidence:
- PR(s): local commit(s) on `main` (Phase 3 Step 1)
- Test run: `pytest tests/flet_redesign -q -p no:asyncio` (60 passed)
- Quality run: `uv run --with pre-commit pre-commit run --all-files` (all hooks passed)

---

## Phase 4: Arena + Charts + Trace

Status: `DN`
Owner: _TBD_
Target Date: _TBD_

- [x] Arena page integrated
- [x] Reward chart integrated
- [x] Regret chart integrated
- [x] Arm pull distribution chart integrated
- [x] Uncertainty/probability view integrated
- [x] Trace table implemented
- [x] Trace export (`json/csv`) implemented
- [x] Replay import validated

Evidence:
- PR(s): local commit(s) on `main` (Phase 4 Step 1)
- Test run: `pytest tests/flet_redesign -q -p no:asyncio` (66 passed)
- Quality run: `uv run --with pre-commit pre-commit run --all-files` (all hooks passed)

---

## Phase 5: Foundation Lessons

Status: `DN`
Owner: _TBD_
Target Date: _TBD_

- [x] Lesson metadata schema implemented
- [x] First five lessons configured
- [x] 5-stage theory card renderer implemented
- [x] Lesson objective evaluator implemented
- [x] Guided control locking implemented
- [x] Step explanation panel integrated
- [x] Lesson completion flow validated

Evidence:
- PR(s): local commit(s) on `main` (Phase 5 Step 1)
- Test run: `pytest tests/flet_redesign -q -p no:asyncio` (71 passed)
- Quality run: `uv run --with pre-commit pre-commit run --all-files` (all hooks passed)

---

## Phase 6: Contextual Lessons

Status: `DN`
Owner: _TBD_
Target Date: _TBD_

- [x] LinUCB implemented
- [x] LinUCB-SW implemented
- [x] Logistic bandit variant implemented
- [x] Feature context inspection panel integrated
- [x] Contextual presets integrated
- [x] Contextual debugger panes integrated
- [x] Formula/regression tests passing

Evidence:
- PR(s): local commit(s) on `main` (Phase 6 Step 1)
- Test run: `pytest tests/flet_redesign -q -p no:asyncio` (81 passed)
- Quality run: `uv run --with pre-commit pre-commit run --all-files` (all hooks passed)

---

## Phase 7: Advanced Discrete + Ensembles

Status: `DN`
Owner: _TBD_
Target Date: _TBD_

- [x] GP-UCB implemented
- [x] Bootstrapped ensemble implemented
- [x] Tree UCB/TS variants implemented
- [x] LinUCB Hybrid implemented
- [x] Capability flag system integrated
- [x] Advanced debugger panes integrated
- [x] Comparative diagnostics added
- [x] Performance baseline re-validated

Evidence:
- PR(s): local commit(s) on `main` (Phase 7 Step 1)
- Test run: `pytest tests/flet_redesign -q -p no:asyncio` (86 passed)
- Quality run: `uv run --with pre-commit pre-commit run --all-files` (all hooks passed)

---

## Phase 8: Continuous + Production Features

Status: `DN`
Owner: _TBD_
Target Date: _TBD_

- [x] CATS policy implemented
- [x] Continuous action controls integrated
- [x] Continuous debugger panes integrated
- [x] Drift detection integrated
- [x] Drift event timeline/indicators integrated
- [x] Checkpoint save/load implemented
- [x] Preset management implemented

Evidence:
- PR(s): local commit(s) on `main` (Phase 8 Step 1)
- Test run: `pytest tests/flet_redesign -q -p no:asyncio` (92 passed)
- Quality run: `uv run --with pre-commit pre-commit run --all-files` (all hooks passed)

---

## Phase 9: Comparison + Sandbox

Status: `IP` (backend partial, UI not started)
Owner: _TBD_
Target Date: _TBD_

### Backend Modules (Existing)

- [x] Multi-policy orchestrator (`comparison/orchestrator.py` — `run_policy_comparison`, `run_batch_comparison`)
- [x] Summary stats (`comparison/stats.py` — `summarize_comparison_runs` with mean/std/CI95)
- [x] Snapshot diff for trace/debugger (`comparison/snapshot_diff.py` — `diff_trace_records`, `diff_debug_snapshots`)
- [x] Sandbox editor (`sandbox.py` — `SandboxEditor` with validation + `build_world_override`)
- [x] Comparison packages (`comparison/__init__.py` — public API)

### UI Pages Required (Not Started)

- [ ] Side-by-side comparison UI page (no file exists)
- [ ] Sandbox editor UI page (no file exists)
- [ ] Snapshot diff viewer component (no file exists)
- [ ] Batch summary stats display (no file exists)

### Integration Required (Not Started)

- [ ] Wire comparison orchestrator into Flet shell
- [ ] Wire sandbox editor into Flet shell
- [ ] Add Comparison route if needed (router has only `/sandbox`)
- [ ] End-to-end flow: select policies → run comparison → display results

### Tests Required (Not Started)

- [ ] Orchestrator deterministic equality tests
- [ ] Stats correctness tests (mean, std, CI95 against known fixtures)
- [ ] Snapshot diff correctness tests
- [ ] Sandbox editor validation tests
- [ ] Comparison + sandbox E2E tests

Evidence:
- PR(s): _TBD_
- Test run: `pytest tests/flet_redesign -q -p no:asyncio` (92 passed, 0 Phase 9 tests exist)

---

## Phase 9.5: Comparison + Sandbox (UI + Tests)

Status: `NS`
Owner: _TBD_
Target Date: _TBD_

This phase was split from Phase 9 after backend modules were discovered to be complete but disconnected from the Flet shell with zero UI integration.

### Deliverables

- [ ] `ui/pages/comparison_page.py` — side-by-side policy comparison page
- [ ] `ui/pages/sandbox_page.py` — open-ended sandbox editor page
- [ ] `ui/components/snapshot_diff_view.py` — diff renderer for trace/debugger snapshots
- [ ] `ui/components/batch_summary_panel.py` — mean/variance/CI95 summary panel
- [ ] Wire comparison orchestrator into shell via `view_models.py` + `main.py`
- [ ] Wire sandbox editor into shell via `view_models.py` + `main.py`
- [ ] Full test suite for Phase 9 + 9.5 modules

### Key Tasks

1. Create `ui/pages/` directory and page modules
2. Build comparison page: policy picker → run → results (charts + table + diff viewer)
3. Build sandbox page: world/param editor → validate → run → results
4. Add `view_models.py` support for comparison/sandbox route models
5. Add `main.py` rendering branches for comparison/sandbox views
6. Add route support (Comparison route) if needed

### Exit Criteria

- Side-by-side comparisons are reproducible and exportable
- Sandbox can create and run custom scenarios without crashes
- All Phase 9 + 9.5 tests pass (target: ~105 total tests)

---

## Phase 10: Polish + Release

Status: `NS`
Owner: _TBD_
Target Date: _TBD_

### UX Polish

- [ ] Spacing and typographic hierarchy pass
- [ ] Interaction transition animations
- [ ] Keyboard navigation support
- [ ] Responsive resize handling
- [ ] Dark mode theme support

### Performance + Reliability

- [ ] Steps/sec benchmark per algorithm class
- [ ] UI update latency profiling
- [ ] Memory baseline for 10k+ step runs
- [ ] Checkpoint integrity under concurrent runs

### Gaps from Earlier Phases

- [ ] Context-free policy debuggers (Random, Epsilon-Greedy, UCB1, Thompson, Softmax — all marked ☐ in algorithm tracker)
- [ ] LinTS policy implementation
- [ ] Drift Detector A implementation
- [ ] Drift Detector B implementation
- [ ] Remaining worlds: ShopSmart, RidePilot, GameBot, LabTrial
- [ ] Integration test suite
- [ ] UI smoke test suite

### Documentation

- [ ] Contributor guide: adding a world
- [ ] Contributor guide: adding a policy
- [ ] Contributor guide: adding a lesson
- [ ] Architecture overview doc
- [ ] Release checklist
- [ ] RC tag

### Exit Criteria

- All release gates pass (see Section 7)
- RC demo scenario succeeds on all target worlds/lessons
- No blocker/high defects open

Evidence:
- PR(s): _TBD_
- Test run: _TBD_

---

## 5) Algorithm Completion Tracker

| Algorithm | Group | Status | Debugger | Tests | Notes |
|---|---|---|---|---|---|
| Random | Context-Free | DN | ☐ | ☑ | Flet redesign baseline policy + tests |
| Epsilon-Greedy | Context-Free | DN | ☐ | ☑ | Phase-1 baseline implementation |
| UCB1 | Context-Free | DN | ☐ | ☑ | Phase-1 baseline implementation |
| Thompson (Bernoulli) | Context-Free | DN | ☐ | ☑ | Phase-1 baseline implementation |
| Softmax | Context-Free | DN | ☐ | ☑ | Phase-1 baseline implementation |
| LinUCB | Linear | DN | ☑ | ☑ | Phase-6 contextual implementation |
| LinUCB-SW | Linear | DN | ☑ | ☑ | Phase-6 contextual implementation |
| LinTS | Linear/Bayesian | NS | ☐ | ☐ | Planned for Phase 10 gap closure |
| Logistic Bandit Variant | Logistic | DN | ☑ | ☑ | Phase-6 contextual implementation |
| GP-UCB | Kernel/Bayesian | DN | ☑ | ☑ | Phase-7 implementation |
| Bootstrapped Ensemble | Ensemble | DN | ☑ | ☑ | Phase-7 implementation |
| LinUCB Hybrid | Hybrid | DN | ☑ | ☑ | Phase-7 implementation |
| RF-UCB | Tree Ensemble | DN | ☑ | ☑ | Implemented as `tree_ucb` bucketized variant |
| RF-TS | Tree Ensemble | DN | ☑ | ☑ | Implemented as `tree_ts` bucketized variant |
| CATS | Continuous | DN | ☑ | ☑ | Phase-8 continuous action implementation |
| Drift Detector A | Drift-Aware | NS | ☐ | ☐ | Planned for Phase 10 gap closure |
| Drift Detector B | Drift-Aware | NS | ☐ | ☐ | Planned for Phase 10 gap closure |

---

## 6) World Integration Tracker

| World | Status | Presets | Lesson Coverage | Notes |
|---|---|---|---|---|
| Rural Clinic | DN | 1 | 3 | + `lesson_linucb` |
| MovieMatch | DN | 1 | 3 | + `lesson_logistic_ucb` |
| NewsFeed | DN | 1 | 2 | + `lesson_linucb_sw` |
| ShopSmart | NS | 0 | 0 | |
| RidePilot | NS | 0 | 0 | |
| GameBot | NS | 0 | 0 | |
| LabTrial | NS | 0 | 0 | |

---

## 7) Quality Gate Tracker

| Gate | Current | Last Run | Owner | Evidence |
|---|---|---|---|---|
| Lint (`ruff`) | DN | 2026-05-24 | _TBD_ | `ruff check src tests` |
| Type checks (`mypy`) | DN | 2026-05-24 | _TBD_ | `uv run --extra dev mypy src/coba` |
| Unit tests (`pytest`) | DN | 2026-05-24 | _TBD_ | `pytest tests/flet_redesign -q` (92 passed, 0 Phase 9 tests) |
| Integration tests | NS | _TBD_ | _TBD_ | No integration test suite exists |
| UI smoke tests | NS | _TBD_ | _TBD_ | No UI smoke test suite exists |
| Deterministic replay check | DN | 2026-05-24 | _TBD_ | `test_replay_payload_*` in `tests/flet_redesign/test_phase1_regression.py` |
| Performance baseline | DN | 2026-05-24 | _TBD_ | `tests/flet_redesign/test_performance_baseline.py` |
| Context-free debugger coverage | NS | _TBD_ | _TBD_ | Random, Epsilon-Greedy, UCB1, Thompson, Softmax — all ☐ |
| World completion (7/7) | 3/7 | _TBD_ | _TBD_ | ShopSmart, RidePilot, GameBot, LabTrial not started |
| Algorithm completion (target 17) | 13/17 | _TBD_ | _TBD_ | LinTS, Drift Detector A/B not started |
| Lesson coverage across worlds | 8/17 | _TBD_ | _TBD_ | 8 lessons configured, 9 remaining (see lesson registry) |

---

## 8) Risks / Blockers Log

| ID | Date | Type | Severity | Status | Owner | Description | Mitigation |
|---|---|---|---|---|---|---|---|
| R-001 | 2026-05-24 | Risk | High | Open | _TBD_ | Phase 9 backend is built but disconnected from Flet shell — no UI pages, no tests, no routing integration | Split Phase 9 into backend (done) + UI/tests (Phase 9.5); prioritize UI page scaffolding |
| R-002 | 2026-05-24 | Risk | Medium | Open | _TBD_ | 4 of 7 worlds not implemented (ShopSmart, RidePilot, GameBot, LabTrial) — 57% world completion | Move remaining worlds to Phase 10 gap closure; not blockers for comparison/sandbox release |
| R-003 | 2026-05-24 | Risk | Medium | Open | _TBD_ | 3 algorithms not implemented (LinTS, Drift Detector A/B) — 76% algorithm completion | Move to Phase 10 gap closure; LinTS is a curriculum dependency, drift detectors are advanced features |
| R-004 | 2026-05-24 | Risk | Medium | Open | _TBD_ | Context-free policies (5 of 13 implemented policies) have no debugger support — violates debugger parity principle | Add debugger renderers in Phase 10 gap closure; low UI impact since debugger is capability-flagged |
| R-005 | 2026-05-24 | Risk | High | Open | _TBD_ | No integration or UI smoke test suites exist — release gate cannot pass without these | Build integration test harness in Phase 9.5; add UI smoke tests in Phase 10 |

---

## 9) Decision Log

| ID | Date | Decision | Rationale | Impacted Areas | Owner |
|---|---|---|---|---|---|
| D-001 | _TBD_ | Example: Discrete-first sequencing | Reduces early complexity | Engine/UI/Curriculum | _TBD_ |

---

## 10) Weekly Update Template

### Week of: _YYYY-MM-DD_

- Overall status: `NS/IP/BL/RV/DN`
- Completed this week:
  - _item_
- In progress:
  - _item_
- Blockers:
  - _item_
- Planned next week:
  - _item_
- Evidence links:
  - PRs:
  - Test runs:
  - Demo notes:

---

## 11) Release Readiness Checklist

- [ ] All milestone gates checked (Phases 0-8: ☑, Phases 9-10: ☐)
- [ ] Phase 9.5 UI + tests complete
- [ ] Phase 10 polish and gap closure complete
- [ ] 13 of 17 target algorithms implemented with debugger + tests (+4 remaining: LinTS, Drift A, Drift B, + context-free debuggers)
- [ ] 3 of 7 worlds integrated and validated (+4 remaining: ShopSmart, RidePilot, GameBot, LabTrial)
- [ ] Curriculum path executable end-to-end (all 8 configured lessons)
- [ ] Integration test suite green
- [ ] UI smoke test suite green
- [ ] Zero open blocker defects
- [ ] Performance baseline within target
- [ ] Final docs updated
- [ ] Release candidate approved
