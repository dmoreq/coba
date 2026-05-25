# COBA Web: System Architecture & Design Decisions

**Tech Stack:** Python 3.10+, Flet 0.85.1, uv

## Architecture Layers

```
src/web/
├── main.py                 # Entry point (ft.app call)
├── app.py                  # AppShell: navigation, theme, routing, view rendering
├── layouts/                # Dashboard layouts (SplitWorkspaceLayout)
├── components/             # Reusable UI widgets (environment, agent, interaction, charts, shared)
├── theme/                  # ColorTokens, ThemeManager (dark/light mode)
├── statemgmt/              # EventBus (pub/sub), InteractionPhase enum
├── ui/                     # View models (frozen dataclasses), charts, preferences, layout specs
├── analysis/               # Metrics, orchestration, stats, diagnostics
├── policies/               # 14 web-facing policy wrappers
├── worlds/                 # 7 narrative simulation worlds + schema
├── curriculum/             # 14 lesson configurations
├── simulator.py            # DiscreteSimulator engine
├── trace.py                # TraceBuffer
├── state.py                # RunConfig, SimulationState, ArmState
├── contracts.py             # BanditPolicy, World, Simulator protocols
├── policy_factory.py       # Policy instantiation
├── policy_capabilities.py  # Static metadata per policy
├── sandbox.py              # SandboxEditor
└── drift_monitor.py        # Drift detection wrapper
```

## Data Flow

```
User clicks Step
  → AppShell._render_view()
    → build_route_ui_model()  (pure dataclass construction)
      → create_world(), build_policy(), build_arena_metrics()
    → SplitWorkspaceLayout.build()  (Flet widget tree)
      → environment, interaction, agent zone components
  → page.update()
```

## Theming

Color tokens are defined in `theme/tokens.py` (40 semantic tokens for light and dark). ThemeManager stores the active set on `page.data`. Every component reads tokens via `ThemeManager.get_tokens(page)`.

## State Management

- EventBus: pub/sub for cross-component events (STEP_COMPLETED, ARM_SELECTED, etc.)
- _SimSession: wraps DiscreteSimulator, RunController, LessonProgressState
- UserPreferences: JSON file-backed persistence
