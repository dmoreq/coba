# Web Module Architecture

Last updated: 2026-05-24

## Overview

`src/web` is a layered, protocol-driven contextual bandit simulation application built on Flet. It contains 7 architectural layers, 15 bandit policies, 7 narrative worlds, 15 interactive lessons, arena analytics, multi-policy comparison, drift detection, and a Flet UI shell.

## Layer Diagram

```
┌─────────────────────────────────────────────────────────┐
│  main.py          (Flet entry point, interactive shell) │  ▸ Presentation
├─────────────────────────────────────────────────────────┤
│  ui/              (View models, controls, pages)         │  ▸ Presentation
├─────────────────────────────────────────────────────────┤
│  curriculum/      (Lesson system, theory cards)          │  ▸ Pedagogy
│  debug/           (Debug pane builders per policy family)│
├─────────────────────────────────────────────────────────┤
│  arena/           (Metrics, run snapshots, diagnostics)  │  ▸ Analytics
│  comparison/      (Orchestrator, stats, snapshot diff)   │
├─────────────────────────────────────────────────────────┤
│  simulator.py     (DiscreteSimulator)                    │  ▸ Engine
│  policy_factory   (build_policy)                         │
│  policy_caps      (Capability registry)                  │
│  preset_manager   (File-backed presets)                  │
├─────────────────────────────────────────────────────────┤
│  policies/        (15 bandit algorithms)                 │  ▸ Domain
│  worlds/          (7 narrative worlds)                   │
│  continuous/      (CATS policy, simulator, world)        │
├─────────────────────────────────────────────────────────┤
│  contracts.py     (Protocols, step result data class)    │  ▸ Contracts
│  state.py         (Run state containers)                 │
│  trace.py         (Trace buffer, serialization)          │
│  drift_monitor    (Drift detection)                      │
│  checkpoint       (Save/load)                            │
└─────────────────────────────────────────────────────────┘
```

## Core Abstractions

### Protocols

- **`BanditPolicy[ArmT, ContextT]`** — `reset()`, `select_arm(context, arms)`, `update(context, arm, reward)`
- **`World[ArmT, ContextT]`** — `reset(seed)`, `get_available_arms()`, `sample_context(step)`, `sample_reward(context, arm)`
- **`DebugSnapshotProvider`** — optional: `get_debug_snapshot() -> dict`
- **`Simulator`** — unified: `reset()`, `step()`, `run_steps(n)`, `replay_payload()`

### Data Models (all frozen dataclasses)

| Model | Purpose |
|---|---|
| `SimulationStepResult` | One step's output |
| `RunConfig` | seed, horizon, autoplay interval |
| `SimulationState` | mutable run state |
| `WorldConfig / ArmDef / FeatureDef` | World schema |
| `TraceBuffer` | append/clear/to_json/to_csv/from_json |
| `ArenaMetrics` | reward/regret/pull counts |
| `RouteUIModel` | View-model for rendering |
| `LessonConfig` | 5-stage theory, objectives, locked controls |

## Data Flow

```
User action → main.py callback
  → RunController.play() / .step()
  → DiscreteSimulator.step()
    → World.sample_context()
    → BanditPolicy.select_arm()
    → World.sample_reward()
    → BanditPolicy.update()
    → SimulationStepResult emitted
    → TraceBuffer.append()
  → _build_view() → build_route_ui_model()
  → _render_shell_view() → Flet controls
  → page.update()
```

## Policy Families

| Family | Policies | Context | Debug |
|---|---|---|---|
| Context-Free | Random, Epsilon-Greedy, UCB1, Thompson, Softmax | No | Yes |
| Linear Contextual | LinUCB, LinUCB-SW, LinTS | Yes | Yes |
| Logistic | Logistic UCB | Yes | Yes |
| Bayesian | GP-UCB | No | Yes |
| Ensemble | Bootstrapped Ensemble | No | Yes |
| Hybrid | LinUCB Hybrid | Yes | Yes |
| Tree Ensemble | Tree UCB, Tree TS | Yes | Yes |
| Continuous | CATS | Yes | Yes |

## Worlds

7 worlds with logistic reward models, 3 arms each, 3 features each (mixed numeric/binary/categorical).

| World | Difficulty | Domain |
|---|---|---|
| Rural Clinic | easy | Healthcare |
| MovieMatch | easy | Streaming |
| NewsFeed | medium | Content ranking |
| ShopSmart | easy | E-commerce |
| RidePilot | medium | Ride-hailing |
| GameBot | medium | Gaming |
| LabTrial | hard | Clinical trials |

## Key Patterns

- **Protocol-Based Polymorphism**: Simulators depend on Protocols, not concrete classes
- **Frozen Data Models**: Immutable dataclasses throughout, except `SimulationState` and `RunController`
- **View-Model Separation**: Pure data models in `ui/` — Flet controls only in `main.py`
- **Factory Pattern**: `build_policy()` and `create_world()` centralize construction
- **Facade Pattern**: `__init__.py` exports ~100 symbols
- **State Machine**: `RunController` = idle → running → paused

## Testing

142 tests across 27 files. Layers: unit (contracts, policies, worlds, state), integration (policy-world loops, lessons, checkpoints), smoke (UI routes, view models, param controls), and specific (comparison, sandbox, debug panes).

## Adding Features

See the contributor guides:
- `docs/contributing/worlds.md` — Adding a world
- `docs/contributing/policies.md` — Adding a policy
- `docs/contributing/lessons.md` — Adding a lesson
