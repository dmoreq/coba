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
| Current Phase | Phase 1 |
| Overall Status | IP |
| Overall Completion | 12% |
| Top Blocker | None |

---

## 3) Milestone Tracker

| Phase | Window | Status | Completion | Owner | Gate |
|---|---|---|---:|---|---|
| Phase 0: Scaffold + Contracts | Day 1 | DN | 100% | _TBD_ | ☑ |
| Phase 1: Discrete Engine + State | Days 2-4 | IP | 0% | _TBD_ | ☐ |
| Phase 2: Core Worlds | Days 5-6 | NS | 0% | _TBD_ | ☐ |
| Phase 3: UI Shell + Controls | Days 7-9 | NS | 0% | _TBD_ | ☐ |
| Phase 4: Arena + Charts + Trace | Days 10-13 | NS | 0% | _TBD_ | ☐ |
| Phase 5: Foundation Lessons | Days 14-18 | NS | 0% | _TBD_ | ☐ |
| Phase 6: Contextual Lessons | Days 19-25 | NS | 0% | _TBD_ | ☐ |
| Phase 7: Advanced Discrete + Ensembles | Days 26-36 | NS | 0% | _TBD_ | ☐ |
| Phase 8: Continuous + Production Features | Days 37-45 | NS | 0% | _TBD_ | ☐ |
| Phase 9: Comparison + Sandbox | Days 46-50 | NS | 0% | _TBD_ | ☐ |
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

Status: `IP`
Owner: _TBD_
Target Date: _TBD_

- [ ] `SimulationState`, `ArmState`, `RunConfig` implemented
- [ ] Trace buffer and serialization implemented
- [ ] Random policy integrated
- [ ] Epsilon-Greedy policy integrated
- [ ] UCB1 policy integrated
- [ ] Thompson Sampling (Beta-Bernoulli) integrated
- [ ] Softmax policy integrated
- [ ] Deterministic seed replay verified
- [ ] Baseline regret tests passing

Evidence:
- PR(s): _TBD_
- Test run: _TBD_

---

## Phase 2: Core Worlds (Clinic, MovieMatch, NewsFeed)

Status: `NS`
Owner: _TBD_
Target Date: _TBD_

- [ ] World schema (`WorldConfig`, `FeatureDef`, `ArmDef`) implemented
- [ ] Validator + parser implemented
- [ ] Rural Clinic world integrated
- [ ] MovieMatch world integrated
- [ ] NewsFeed world integrated
- [ ] Difficulty presets per world
- [ ] World switching UI/API connected
- [ ] World fixtures committed

Evidence:
- PR(s): _TBD_
- Test run: _TBD_

---

## Phase 3: UI Shell + Controls

Status: `NS`
Owner: _TBD_
Target Date: _TBD_

- [ ] Three-pane layout implemented
- [ ] `ScenePanel` component implemented
- [ ] `TreatmentCard` component implemented
- [ ] Run controls implemented (`step/play/pause/reset`)
- [ ] Param controls rendered from `ParamControlSpec`
- [ ] Tooltip component implemented with formula/intuition guidance
- [ ] UI state persistence (local preferences)
- [ ] Desktop layout sanity check passed

Evidence:
- PR(s): _TBD_
- Test run: _TBD_

---

## Phase 4: Arena + Charts + Trace

Status: `NS`
Owner: _TBD_
Target Date: _TBD_

- [ ] Arena page integrated
- [ ] Reward chart integrated
- [ ] Regret chart integrated
- [ ] Arm pull distribution chart integrated
- [ ] Uncertainty/probability view integrated
- [ ] Trace table implemented
- [ ] Trace export (`json/csv`) implemented
- [ ] Replay import validated

Evidence:
- PR(s): _TBD_
- Test run: _TBD_

---

## Phase 5: Foundation Lessons

Status: `NS`
Owner: _TBD_
Target Date: _TBD_

- [ ] Lesson metadata schema implemented
- [ ] First five lessons configured
- [ ] 5-stage theory card renderer implemented
- [ ] Lesson objective evaluator implemented
- [ ] Guided control locking implemented
- [ ] Step explanation panel integrated
- [ ] Lesson completion flow validated

Evidence:
- PR(s): _TBD_
- Test run: _TBD_

---

## Phase 6: Contextual Lessons

Status: `NS`
Owner: _TBD_
Target Date: _TBD_

- [ ] LinUCB implemented
- [ ] LinUCB-SW implemented
- [ ] Logistic bandit variant implemented
- [ ] Feature context inspection panel integrated
- [ ] Contextual presets integrated
- [ ] Contextual debugger panes integrated
- [ ] Formula/regression tests passing

Evidence:
- PR(s): _TBD_
- Test run: _TBD_

---

## Phase 7: Advanced Discrete + Ensembles

Status: `NS`
Owner: _TBD_
Target Date: _TBD_

- [ ] GP-UCB implemented
- [ ] Bootstrapped ensemble implemented
- [ ] Tree UCB/TS variants implemented
- [ ] LinUCB Hybrid implemented
- [ ] Capability flag system integrated
- [ ] Advanced debugger panes integrated
- [ ] Comparative diagnostics added
- [ ] Performance baseline re-validated

Evidence:
- PR(s): _TBD_
- Test run: _TBD_

---

## Phase 8: Continuous + Production Features

Status: `NS`
Owner: _TBD_
Target Date: _TBD_

- [ ] CATS policy implemented
- [ ] Continuous action controls integrated
- [ ] Continuous debugger panes integrated
- [ ] Drift detection integrated
- [ ] Drift event timeline/indicators integrated
- [ ] Checkpoint save/load implemented
- [ ] Preset management implemented

Evidence:
- PR(s): _TBD_
- Test run: _TBD_

---

## Phase 9: Comparison + Sandbox

Status: `NS`
Owner: _TBD_
Target Date: _TBD_

- [ ] Multi-policy orchestrator implemented
- [ ] Side-by-side comparison UI integrated
- [ ] Batch repeated-seed execution integrated
- [ ] Summary stats with variance/confidence integrated
- [ ] Sandbox editor integrated
- [ ] Snapshot diff for trace/debugger integrated

Evidence:
- PR(s): _TBD_
- Test run: _TBD_

---

## Phase 10: Polish + Release

Status: `NS`
Owner: _TBD_
Target Date: _TBD_

- [ ] UX polish pass complete
- [ ] Performance profile + fixes complete
- [ ] Accessibility/interaction sanity pass complete
- [ ] Contributor extension docs complete
- [ ] Full regression suite green
- [ ] Release candidate tag prepared

Evidence:
- PR(s): _TBD_
- Test run: _TBD_

---

## 5) Algorithm Completion Tracker

| Algorithm | Group | Status | Debugger | Tests | Notes |
|---|---|---|---|---|---|
| Random | Context-Free | NS | ☐ | ☐ | |
| Epsilon-Greedy | Context-Free | NS | ☐ | ☐ | |
| UCB1 | Context-Free | NS | ☐ | ☐ | |
| Thompson (Bernoulli) | Context-Free | NS | ☐ | ☐ | |
| Softmax | Context-Free | NS | ☐ | ☐ | |
| LinUCB | Linear | NS | ☐ | ☐ | |
| LinUCB-SW | Linear | NS | ☐ | ☐ | |
| LinTS | Linear/Bayesian | NS | ☐ | ☐ | |
| Logistic Bandit Variant | Logistic | NS | ☐ | ☐ | |
| GP-UCB | Kernel/Bayesian | NS | ☐ | ☐ | |
| Bootstrapped Ensemble | Ensemble | NS | ☐ | ☐ | |
| LinUCB Hybrid | Hybrid | NS | ☐ | ☐ | |
| RF-UCB | Tree Ensemble | NS | ☐ | ☐ | |
| RF-TS | Tree Ensemble | NS | ☐ | ☐ | |
| CATS | Continuous | NS | ☐ | ☐ | |
| Drift Detector A | Drift-Aware | NS | ☐ | ☐ | |
| Drift Detector B | Drift-Aware | NS | ☐ | ☐ | |

---

## 6) World Integration Tracker

| World | Status | Presets | Lesson Coverage | Notes |
|---|---|---|---|---|
| Rural Clinic | NS | 0 | 0 | |
| MovieMatch | NS | 0 | 0 | |
| NewsFeed | NS | 0 | 0 | |
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
| Unit tests (`pytest`) | DN | 2026-05-24 | _TBD_ | `pytest tests/flet_redesign -q` |
| Integration tests | NS | _TBD_ | _TBD_ | |
| UI smoke tests | NS | _TBD_ | _TBD_ | |
| Deterministic replay check | NS | _TBD_ | _TBD_ | |
| Performance baseline | NS | _TBD_ | _TBD_ | |

---

## 8) Risks / Blockers Log

| ID | Date | Type | Severity | Status | Owner | Description | Mitigation |
|---|---|---|---|---|---|---|---|
| R-001 | _TBD_ | Risk | Medium | Open | _TBD_ | Example: debugger complexity growth | Use capability flag + renderer registry |

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

- [ ] All milestone gates checked
- [ ] All required algorithms implemented with debugger + tests
- [ ] All seven worlds integrated and validated
- [ ] Curriculum path executable end-to-end
- [ ] Zero open blocker defects
- [ ] Performance baseline within target
- [ ] Final docs updated
- [ ] Release candidate approved
