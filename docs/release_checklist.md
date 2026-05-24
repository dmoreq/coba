# Release Checklist

Last updated: 2026-05-24

## Pre-Release Gates

- [x] Lint: `ruff check src/web tests/flet_redesign` — passes
- [x] Type check: `mypy src/coba src/web` — passes
- [x] Unit tests: `pytest tests/flet_redesign -q` — 142 passed
- [x] Integration tests: policy-world loops, lesson progression, checkpoint roundtrip
- [x] UI smoke tests: route models, shell stack, navigation, param controls
- [x] Comparison tests: orchestrator determinism, stats accuracy, snapshot diff
- [x] Sandbox tests: editor validation, world overrides, immutability
- [x] Debug pane tests: all 7 builder families, context-free through continuous
- [x] Deterministic replay: `test_replay_payload_*` in `test_phase1_regression.py`
- [x] Performance baseline: `test_performance_baseline.py`

## Feature Completeness

- [x] 15 bandit policies (target: 17 — LinTS added, drift detectors deferred)
- [x] 7 narrative worlds (target: 7)
- [x] 15 curriculum lessons (target: 8+)
- [x] All policies implement `DebugSnapshotProvider`
- [x] Debug pane builders for all 8 policy families
- [x] Interactive simulation shell (Play/Pause/Step/Reset with live state)
- [x] Lesson progression with 5-stage theory cards and objective evaluation
- [x] Parameter controls with stage-locked progressive disclosure
- [x] Side-by-side policy comparison with batch stats (mean/std/CI95)
- [x] Sandbox editor with world overrides and validation
- [x] Snapshot diff for trace and debug state
- [x] Comparison route with navigation
- [x] Chart data generation
- [x] Trace table model
- [x] Preferences persistence
- [x] Checkpoint save/load
- [x] Drift detection (CUSUM + ADWIN)
- [x] Continuous-action simulation with regret and replay
- [x] Configurable continuous world from WorldConfig

## Documentation

- [x] Architecture overview: `docs/web_architecture.md`
- [x] Contributor guide — worlds: `docs/contributing/worlds.md`
- [x] Contributor guide — policies: `docs/contributing/policies.md`
- [x] Contributor guide — lessons: `docs/contributing/lessons.md`
- [x] Implementation plan: `docs/web_implementation_plan.md`
- [x] Progress tracker: `docs/flet_redesign_progress_tracker.md`
- [x] Original implementation plan: `docs/flet_redesign_implementation_plan.md`
- [x] Release checklist: `docs/release_checklist.md` (this file)

## Known Gaps (Post-RC)

- 2 drift detector variants (A/B) implemented as CUSUM+ADWIN (in `web.drift`)
- LinTS policy implemented and registered
- No native mobile targets
- Charts render as text summaries; full Flet chart rendering needs browser canvas

## RC Tag Instructions

```bash
git add -A
git commit -m "Release candidate: 15 policies, 7 worlds, 15 lessons, interactive shell

- Interactive Flet shell with Play/Pause/Step/Reset
- Side-by-side policy comparison with stats
- Sandbox editor with world overrides
- 5-stage lesson pedagogy with objective evaluation
- Full debugger coverage across all policy families
- Continuou-saction simulation with regret tracking
- 142 tests passing

Co-authored-by: CommandCodeBot <noreply@commandcode.ai>"
git tag v0.1.0-rc1
```

## Verification

```bash
# Full test suite
pytest tests/flet_redesign -q -p no:asyncio

# Start the app
PYTHONPATH=src python -c "from web import run; run()"

# Demo scenario: run all policy-world pairs
PYTHONPATH=src python -c "
from web import build_policy, DiscreteSimulator, RunConfig, create_world
for world_id in ['rural_clinic','moviematch','newsfeed']:
    w = create_world(world_id)
    for pid in ['random','ucb1','epsilon_greedy']:
        p = build_policy(pid, seed=0)
        sim = DiscreteSimulator(p, w, RunConfig(seed=0, horizon=100))
        sim.reset(); sim.run_steps(100)
        print(f'{world_id}/{pid}: reward={sim.state.cumulative_reward:.1f}')
"
```
