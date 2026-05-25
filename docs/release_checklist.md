# Release Checklist

Last updated: 2026-05-24

## Pre-Release Gates

- [x] Lint: `ruff check src/web tests/flet_redesign` — passes
- [x] Type check: `mypy src/coba src/web` — passes
- [x] Unit tests: `pytest tests/flet_redesign tests/web -q -p no:asyncio` — passes
- [x] Full suite: `pytest tests/ -p no:asyncio --ignore=tests/test_shared_sim.py` — passes
- [x] Integration tests: policy-world loops, lesson progression
- [x] UI smoke tests: route models, navigation, param controls
- [x] Comparison tests: orchestrator determinism, stats accuracy
- [x] Sandbox tests: editor validation, world overrides, immutability
- [x] Debug pane tests: all 7 builder families
- [x] Deterministic replay
- [x] Edge case tests: 45+ tests covering race conditions, invalid inputs, boundaries
- [x] Theme tests: WCAG contrast ratio ≥ 4.5, token field parity, spacing/font ordering
- [x] Event bus tests: subscribe, emit, unsubscribe, error isolation, clear, async

## Feature Completeness

- [x] 15 bandit policies
- [x] 7 narrative worlds
- [x] 14 curriculum lessons
- [x] Interactive simulation shell (Play/Pause/Step/Reset with live state)
- [x] Lesson progression with 5-stage theory cards and objective evaluation
- [x] Parameter controls with stage-locked progressive disclosure
- [x] Side-by-side policy comparison with batch stats (mean/std/CI95)
- [x] Sandbox editor with world overrides and validation
- [x] Preferences persistence
- [x] Continuous-action simulation with regret and replay
- [x] Configurable continuous world from WorldConfig
- [x] Theme system (ColorTokens, dark/light mode)
- [x] SplitWorkspaceLayout (3-zone dashboard)
- [x] EventBus (pub/sub for cross-component communication)
- [x] Environment/agent/interaction/chart components with theme-aware styling

## Documentation

- [x] Architecture overview: `docs/ARCHITECTURE.md`
- [x] Contributor guide — worlds: `docs/contributing/worlds.md`
- [x] Contributor guide — policies: `docs/contributing/policies.md`
- [x] Contributor guide — lessons: `docs/contributing/lessons.md`
- [x] Release checklist: `docs/release_checklist.md` (this file)

## Known Gaps

- Chart components use ft.Container bars (Flet 0.85.1 lacks LineChart/BarChart)
- No test coverage for Flet widget builders (6 component files, layouts, app.py)
- Policy base classes not yet extracted (14 policy wrappers duplicate _ensure_arms)

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
