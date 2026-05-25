# Option C: Split-Workspace Dashboard — Implementation Plan

> **Architect:** Command Code (Plan Mode)
> **Date:** 2025-05-25
> **Based on:** Thorough exploration of all 79 files in `src/web/`, 32 test files, and the full VM+P data pipeline.

---

## 1. What We Found in the Codebase

### Architecture
- **VM+P pattern**: 19 frozen dataclasses as view-models → pure data → Flet widgets. Clean 3-layer separation.
- **main.py** (661 lines): The monolith — 18 functions mixing rendering, state management, event handling, and autoplay.
- **No charts**: Chart data exists (`ChartData`, `ArenaMetrics`, `SeriesPoint`) but renders as plain `ft.Text`. No `ft.LineChart`/`ft.BarChart` anywhere.
- **No theming**: All colors/styles are hardcoded inline hex values. Zero dark/light mode support.
- **No animation**: Steps happen silently. No visual feedback on arm selection, reward, or knowledge updates.
- **State**: Module-level globals (`_page`, `_session`, `_pref_store`, `_autoplay_task`). No pub/sub.

### Dead Code (~455 lines to delete)
| File | Why |
|------|-----|
| `drift/` (3 files, 155L) | Local CUSUM/ADWIN — zero callers. Drift uses `coba.drift.PageHinkleyDetector`. |
| `shell.py` (45L) | `ShellView`/`build_shell_stack` — never called. `main.py` has its own `_render_shell_view`. |
| `state_store.py` (40L) | `AppStateStore` — superseded by `PreferencesStore` + `_SimSession`. |
| `preset_manager.py` (45L) | `PresetManager` — never instantiated. |
| `checkpoint.py` (55L) | `save_checkpoint`/`load_checkpoint` — zero callers. `TraceBuffer` handles serialization. |
| `debug/context_free.py` (45L) | `build_cf_debug_pane` — never called. |
| `debug/continuous.py` (20L) | Token `build_continuous_debug_pane` — merge into `debug/advanced.py`. |
| `comparison/snapshot_diff.py` (40L) | `diff_trace_records()` / `diff_debug_snapshots()` — never called. Move `SnapshotDiffResult` dataclass into the view file. |

### Redundant/Duplicated Logic (~800 lines to refactor)
| Area | Issue | Savings |
|------|-------|---------|
| `worlds/base.py` + `continuous/configurable_world.py` | ~65 lines verbatim copy: `_sample_feature`, `_expected_probability`, `sample_context`, `reset` | Extract `_BaseWorld` → both inherit |
| `simulator.py` + `continuous/simulator.py` | Same `step()`, `run_steps()`, `reset()` pattern. `ContinuousSimulator` not even wired in. | `ContinuousSimulator` extends `DiscreteSimulator` |
| Policy files (14 total) | `_ensure_arms`, `_reward_sum`/`_pulls`, `get_debug_snapshot` duplicated across context-free policies (5 files, ~360L) | Extract `ContextFreePolicyBase` |
| `linucb_policy.py` + `linucb_sw_policy.py` | 70% shared structure. SW is LinUCB + deque window. | SW extends LinUCB |
| `tree_ucb_policy.py` + `tree_ts_policy.py` | `_bucket()` method character-for-character identical. | Extract `BucketPolicyBase` |
| `ALL_POLICY_IDS` tuple | Defined 3 times: `comparison_page.py:18`, `sandbox_page.py:37`, and implicitly in `main.py` selector | Move to `policy_capabilities.py` |
| `view_models.py` + `sandbox_page.py` + `main.py` | World-creation logic (create_world, seed=0, get_world_config) lives in 3 places | Extract `build_simulator()` factory |
| `arena/` + `comparison/` | Two packages with overlapping concerns | Merge into `analysis/` |

---

## 2. What We're Building (Option C Layout)

```
┌──────────────────────────────────────────────────────────────────┐
│ TOP BAR: Theme toggle · World selector · Policy selector · Speed │
│ COBA · RidePilot | ε-Greedy | Step 12/100                       │
├────────────┬─────────────────────────┬───────────────────────────┤
│ ENVIRONMENT│    INTERACTION LOOP     │         AGENT              │
│ (Teal tint)│    (Neutral bridge)     │     (Amber tint)           │
│            │                         │                            │
│ World Card │  ① Context ────────▶   │  Knowledge Table           │
│ Context:   │                         │  Priority   ~0.72 (8)     │
│  Time: 5pm │  ② ◀─── Arm Pick ────  │  Standard   ~0.45 (3)     │
│  Rain: Yes │                         │  Pool       ~0.38 (1)     │
│  Demand:H  │  ③ Reward ────────▶    │                            │
│            │                         │  Policy State Card         │
│ ┌────────┐ │  ④ ◀── Update Model ── │  ε = 0.1 | α = 1.0        │
│ │Hidden  │ │                         │                            │
│ │Truth 🔒│ │  [Step ▸] [▶ Play]      │  Pull Histogram (compact)  │
│ │[sliders]│ │  [↺ Reset] [⏩ Run 50]  │                            │
│ └────────┘ │                         │                            │
├────────────┴─────────────────────────┴───────────────────────────┤
│ CHARTS ZONE ▾ (collapsible)                                      │
│ ┌──────────────────────────┐  ┌─────────────────────────────────┐│
│ │ Cumulative Regret        │  │ Arm Selection Histogram         ││
│ │ ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄ │  │ Priority ██████████████          ││
│ │ (animated LineChart)     │  │ Standard ██████                 ││
│ └──────────────────────────┘  └─────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
```

### How the 4-Phase Interaction Loop Works

Each step triggers an animated sequence:

| Phase | Visual | Duration |
|-------|--------|----------|
| ① Context Generated | New context values slide in (teal pulse), teal arrow animates right | 300ms |
| ② Arm Selected | Selected arm card grows 105%→100%, amber glow border, amber arrow points left | 400ms |
| ③ Reward Received | Green ✓ pulse (success) or red shake (failure) overlay, green/red arrow | 600ms |
| ④ Knowledge Updated | Changed cells in KnowledgeTable flash amber, settle to new values | 300ms |

Speed slider (0.25× to 4×) multiplies all delays. Run-N skips animation entirely.

---

## 3. Final Folder Structure

```
src/web/
├── main.py                          # Thin entry point (ft.app call only)
├── app.py                           # AppShell: root View, theme wiring, nav rail, routing
├── router.py                        # Route enum/spec/resolution (keep, minor: remove shell refs)
│
├── theme/                           # 🌗 NEW: Theming system
│   ├── __init__.py
│   ├── tokens.py                    # ColorTokens dataclass, LIGHT_TOKENS, DARK_TOKENS
│   ├── theme_manager.py             # Applies ft.Theme to page, mode toggle helper
│   └── constants.py                 # Spacing scale, typography scale, animation durations
│
├── state/                           # 🧠 NEW: Centralized state + event bus
│   ├── __init__.py
│   ├── app_state.py                 # AppState dataclass (replaces _SimSession globals)
│   ├── simulation_controller.py     # Wraps DiscreteSimulator, step animation, run-n
│   ├── interaction_state.py         # InteractionPhase enum, per-step animation state
│   └── event_bus.py                 # Simple pub/sub for cross-component events
│
├── layouts/                         # 📐 NEW: Layout system
│   ├── __init__.py
│   ├── base.py                      # BaseLayout protocol
│   └── split_workspace.py           # Option C: 3-zone + bottom charts layout
│
├── components/                      # 🧩 NEW: Reusable Flet UI components
│   ├── __init__.py
│   ├── shared/                      # Shared/common
│   │   ├── __init__.py
│   │   ├── section_header.py        # Zone header: icon + title + accent color strip
│   │   ├── metric_badge.py          # Compact value + label display
│   │   └── empty_state.py           # "Nothing to show" placeholder
│   │
│   ├── environment/                 # 🌊 Environment zone (teal)
│   │   ├── __init__.py
│   │   ├── world_card.py            # World name, description, illustration
│   │   ├── context_display.py       # Feature vector as styled key-value cards
│   │   └── hidden_truth_panel.py    # Collapsible ground-truth probability sliders
│   │
│   ├── agent/                       # 🟠 Agent zone (amber)
│   │   ├── __init__.py
│   │   ├── knowledge_table.py       # Arms × mean reward × pulls DataTable
│   │   ├── pull_counter.py          # Compact horizontal bar display
│   │   ├── uncertainty_display.py   # Confidence interval visualization
│   │   └── policy_state_card.py     # Algorithm params display (ε, α, etc.)
│   │
│   ├── interaction/                 # ⚡ Interaction zone (neutral)
│   │   ├── __init__.py
│   │   ├── loop_visualizer.py       # 4-phase animated bridge with directional arrows
│   │   ├── arm_cards.py             # Arm selection cards with glow animation
│   │   ├── reward_feedback.py       # Green/red pulse overlay for reward
│   │   └── step_indicator.py        # Current step / total progress bar
│   │
│   ├── controls/                    # 🎮 Control components
│   │   ├── __init__.py
│   │   ├── step_controls.py         # Step, Play/Pause, Reset, Run-N buttons
│   │   ├── speed_slider.py          # Animation speed slider
│   │   ├── world_selector.py        # World dropdown
│   │   ├── policy_selector.py       # Policy dropdown
│   │   └── theme_toggle.py          # Dark/light mode switch
│   │
│   └── charts/                      # 📊 Chart components
│       ├── __init__.py
│       ├── chart_theme.py           # Theme-aware chart styling from ColorTokens
│       ├── regret_chart.py          # Cumulative regret ft.LineChart (animated)
│       ├── arm_histogram.py         # Arm selection ft.BarChart (animated)
│       ├── reward_timeline.py       # Per-step reward sparkline
│       └── knowledge_heatmap.py     # Arm × Context grid (for contextual bandits)
│
├── ui/                              # 📋 View models (keep architecture)
│   ├── __init__.py
│   ├── view_models.py               # RouteUIModel, build_route_ui_model (extend)
│   ├── charts.py                     # ChartData (keep, consumed by ft.LineChart now)
│   ├── layout.py                     # Layout specs (extend for split-workspace)
│   ├── param_controls.py             # ParamControlSpec (extend: prob sliders)
│   ├── preferences.py                # UserPreferences, PreferencesStore (keep)
│   ├── run_controls.py               # RunController (keep)
│   └── models.py                     # 🆕 Consolidated thin dataclasses:
│       ├── ScenePanelModel           # (was scene_panel.py)
│       ├── TreatmentCardModel        # (was treatment_card.py)
│       ├── TraceTableModel           # (keep)
│       ├── LessonPanelModel          # (was lesson_models.py)
│       ├── ContextInspectionModel    # (was context_inspection.py)
│       ├── ParamTooltip              # (was tooltips.py)
│       ├── DiffViewProps             # (was snapshot_diff_view.py, simplified)
│       └── BatchSummaryPanelProps    # (keep)
│
├── analysis/                         # 🆕 Merged arena/ + comparison/
│   ├── __init__.py
│   ├── metrics.py                    # build_arena_metrics (was arena/metrics.py)
│   ├── orchestrator.py               # run_policy_comparison (was comparison/orchestrator.py)
│   ├── stats.py                      # summarize_comparison_runs (was comparison/stats.py)
│   └── diagnostics.py                # compute_comparison_diagnostics (was arena/diagnostics.py)
│
├── policies/                         # 🔧 Refactored policy hierarchy
│   ├── __init__.py
│   ├── base.py                       # 🆕 ContextFreePolicyBase, LinearContextualPolicyBase
│   ├── random_policy.py              #    (extends ContextFreePolicyBase, ~15L of unique logic)
│   ├── epsilon_greedy_policy.py      #    (extends ContextFreePolicyBase, ~20L)
│   ├── softmax_policy.py             #    (extends ContextFreePolicyBase, ~25L)
│   ├── thompson_policy.py            #    (extends ContextFreePolicyBase, ~25L)
│   ├── ucb1_policy.py                #    (extends ContextFreePolicyBase, ~25L)
│   ├── lints_policy.py               #    (extends LinearContextualPolicyBase)
│   ├── linucb_policy.py              #    (extends LinearContextualPolicyBase)
│   ├── linucb_sw_policy.py           #    (extends LinUCBPolicy, overrides window)
│   ├── linucb_hybrid_policy.py       #    (keep, unique structure)
│   ├── logistic_ucb_policy.py        #    (keep)
│   ├── gp_ucb_policy.py              #    (keep)
│   ├── tree_base.py                  # 🆕 BucketPolicyBase
│   ├── tree_ucb_policy.py            #    (extends BucketPolicyBase, ~15L unique)
│   ├── tree_ts_policy.py             #    (extends BucketPolicyBase, ~15L unique)
│   ├── bootstrapped_ensemble_policy.py # (keep)
│   └── contextual_utils.py           #    context_to_vector (inline into __init__.py)
│
├── worlds/                           # Keep, minor refactor
│   ├── __init__.py
│   ├── base.py                       # Extract _BaseWorld, ConfigurableWorld extends it
│   ├── schema.py                     # Keep
│   ├── registry.py                   # Keep
│   ├── presets.py                    # Keep
│   └── core_worlds.py               # Keep
│
├── simulator.py                      # Keep (DiscreteSimulator)
├── trace.py                          # Keep (TraceBuffer)
├── state.py                          # Keep (RunConfig, SimulationState, ArmState)
├── contracts.py                      # Keep (Protocols)
├── policy_factory.py                 # 🆕 Add ALL_POLICY_IDS, POLICY_LABELS here
├── policy_capabilities.py            # Keep, add shared constants
├── drift_monitor.py                  # Keep (wraps coba.drift)
├── sandbox.py                        # Keep (SandboxEditor)
│
├── curriculum/                       # Keep
│   ├── __init__.py
│   └── lessons.py                    # (keep, 14 lesson configs)
│
├── continuous/                       # Minor refactor
│   ├── __init__.py
│   ├── simulator.py                  # Extend DiscreteSimulator
│   ├── configurable_world.py         # Extend _BaseWorld from worlds/base.py
│   ├── schemas.py                    # Keep
│   └── cats_policy.py               # Keep
│
└── debug/                            # Cleaned up
    ├── __init__.py
    ├── advanced.py                   # Merge continuous.py into here
    └── contextual.py                 # Keep
```

### Files to DELETE
```
src/web/shell.py
src/web/state_store.py
src/web/preset_manager.py
src/web/checkpoint.py
src/web/drift/                        # Entire directory (3 files)
src/web/debug/context_free.py
src/web/debug/continuous.py           # Logic merged into advanced.py
src/web/comparison/snapshot_diff.py   # Dataclass moved to ui/models.py
src/web/ui/tooltips.py                # Merged into ui/models.py
src/web/ui/context_inspection.py      # Merged into ui/models.py
src/web/ui/lesson_models.py           # Merged into ui/models.py
src/web/ui/components/scene_panel.py  # Merged into ui/models.py
src/web/ui/components/treatment_card.py # Merged into ui/models.py
src/web/ui/components/snapshot_diff_view.py # Merged into ui/models.py
src/web/arena/                        # Merged into analysis/
src/web/comparison/                   # Merged into analysis/ (except snapshot_diff.py which is deleted)
src/web/ui/pages/comparison_page.py   # Consolidated into ui/view_models.py
src/web/ui/pages/sandbox_page.py      # Consolidated into ui/view_models.py
```

---

## 4. Implementation Phases (with exact file operations)

### Phase 1: Clean Slate — Delete Dead Code & Consolidate (No behavior changes)

**Goal:** Remove 455 lines of dead code. Consolidate thin dataclass files. Merge arena/comparison into analysis/. Extract policy base classes. Zero behavior change — all tests still pass.

#### Step 1.1: Delete dead code
```
DELETE  src/web/shell.py
DELETE  src/web/state_store.py
DELETE  src/web/preset_manager.py
DELETE  src/web/checkpoint.py
DELETE  src/web/drift/__init__.py
DELETE  src/web/drift/adwin_detector.py
DELETE  src/web/drift/cusum_detector.py
DELETE  src/web/debug/context_free.py
DELETE  src/web/debug/continuous.py         (inline build_continuous_debug_pane → advanced.py)
DELETE  src/web/comparison/snapshot_diff.py  (move SnapshotDiffResult → ui/models.py)
DELETE  src/web/__init__.py references to deleted modules
```

#### Step 1.2: Consolidate thin dataclass files into `ui/models.py`
```
CREATE  src/web/ui/models.py
  ← src/web/ui/components/scene_panel.py      (ScenePanelModel)
  ← src/web/ui/components/treatment_card.py   (TreatmentCardModel)
  ← src/web/ui/components/trace_table.py      (TraceTableModel + builder)
  ← src/web/ui/components/batch_summary_panel.py (BatchSummaryPanelProps + builder)
  ← src/web/ui/components/snapshot_diff_view.py (DiffViewProps, simplified builder — remove dead two-param branch)
  ← src/web/ui/lesson_models.py               (LessonPanelModel)
  ← src/web/ui/context_inspection.py           (ContextInspectionModel)
  ← src/web/ui/tooltips.py                     (ParamTooltip)
  ← SnapshotDiffResult from comparison/snapshot_diff.py
DELETE  src/web/ui/components/scene_panel.py
DELETE  src/web/ui/components/treatment_card.py
DELETE  src/web/ui/components/trace_table.py
DELETE  src/web/ui/components/batch_summary_panel.py
DELETE  src/web/ui/components/snapshot_diff_view.py
DELETE  src/web/ui/lesson_models.py
DELETE  src/web/ui/context_inspection.py
DELETE  src/web/ui/tooltips.py
  UPDATE src/web/ui/components/__init__.py → re-export from models.py
  UPDATE all imports across the codebase
```

#### Step 1.3: Merge arena/ + comparison/ → analysis/
```
CREATE  src/web/analysis/__init__.py
CREATE  src/web/analysis/metrics.py           ← src/web/arena/metrics.py
CREATE  src/web/analysis/diagnostics.py       ← src/web/arena/diagnostics.py
CREATE  src/web/analysis/orchestrator.py      ← src/web/comparison/orchestrator.py
CREATE  src/web/analysis/stats.py             ← src/web/comparison/stats.py
DELETE  src/web/arena/                        (entire directory)
DELETE  src/web/comparison/                   (entire directory)
  UPDATE all imports
```

#### Step 1.4: Extract policy base classes (refactor, zero behavior change)
```
CREATE  src/web/policies/base.py
  Extract ContextFreePolicyBase: _ensure_arms, _reward_sum, _pulls tracking, get_debug_snapshot shape
  Extract LinearContextualPolicyBase: add feature matrix + arm weight maintenance
  Extract BucketPolicyBase: _bucket, _ensure_arms shared by tree_ucb/ts

UPDATE  src/web/policies/random_policy.py       → extends ContextFreePolicyBase
UPDATE  src/web/policies/epsilon_greedy_policy.py → extends ContextFreePolicyBase
UPDATE  src/web/policies/softmax_policy.py      → extends ContextFreePolicyBase
UPDATE  src/web/policies/thompson_policy.py     → extends ContextFreePolicyBase
UPDATE  src/web/policies/ucb1_policy.py         → extends ContextFreePolicyBase
UPDATE  src/web/policies/lints_policy.py        → extends LinearContextualPolicyBase
UPDATE  src/web/policies/linucb_policy.py       → extends LinearContextualPolicyBase
UPDATE  src/web/policies/linucb_sw_policy.py    → extends LinUCBPolicy (override window)
UPDATE  src/web/policies/tree_ucb_policy.py     → extends BucketPolicyBase
UPDATE  src/web/policies/tree_ts_policy.py      → extends BucketPolicyBase
```

#### Step 1.5: Consolidate shared constants
```
UPDATE  src/web/policy_capabilities.py
  ← ALL_POLICY_IDS tuple (from comparison_page.py + sandbox_page.py)
  ← POLICY_LABELS dict (from main.py _render_policy_selector)

UPDATE  src/web/policy_factory.py
  Export ALL_POLICY_IDS, POLICY_LABELS

UPDATE  src/web/ui/view_models.py → import ALL_POLICY_IDS from policy_capabilities
UPDATE  src/web/main.py → import POLICY_LABELS from policy_capabilities
```

#### Step 1.6: Extract world simulator factory
```
CREATE  src/web/simulator_factory.py
  build_simulator(world_id, policy_id, seed, horizon) → DiscreteSimulator
  Used by: main.py _reset_simulator, view_models.py build_route_ui_model, sandbox_page.py build_sandbox_model

UPDATE  3 call sites to use the factory
```

#### Step 1.7: Extract shared world base
```
UPDATE  src/web/worlds/base.py
  Extract _BaseWorld class with shared _sample_feature, _expected_probability, sample_context, reset
  ConfigurableWorld extends _BaseWorld

UPDATE  src/web/continuous/configurable_world.py
  ContinuousWorld extends _BaseWorld (only overrides sample_reward + optimal logic)
```

**Verification checkpoint:** All existing tests pass. `make test && make test-web` green.

**Commit:** `refactor(web): delete dead code, consolidate dataclasses, extract policy bases and shared factories`

---

### Phase 2: Theme System & Layout Foundation

**Goal:** Color tokens system, theme manager, dark/light mode toggle, SplitWorkspaceLayout rendering with placeholder content.

#### Step 2.1: Create theme system
```
CREATE  src/web/theme/__init__.py
CREATE  src/web/theme/tokens.py
  ColorTokens frozen dataclass with all semantic fields
  LIGHT_TOKENS, DARK_TOKENS module-level constants

CREATE  src/web/theme/theme_manager.py
  ThemeManager.apply_theme(page, mode): builds ft.Theme, stores tokens in page.session
  ThemeManager.get_tokens(page): returns ColorTokens from session
  ThemeManager.toggle(page): toggles between light/dark

CREATE  src/web/theme/constants.py
  SpacingScale: XS=4, SM=8, MD=12, LG=20, XL=32
  FontScale: BODY=14, SMALL=12, CAPTION=10, HEADING=24, TITLE=18
  AnimationDurations: PHASE_CONTEXT=300, PHASE_ARM=400, PHASE_REWARD=600, PHASE_KNOWLEDGE=300, CHART=300
```

#### Step 2.2: Create shared components
```
CREATE  src/web/components/shared/__init__.py
CREATE  src/web/components/shared/section_header.py
  Builds zone header: icon + title text + accent color strip below
  Reads tokens from ThemeManager.get_tokens(page)

CREATE  src/web/components/shared/metric_badge.py
  Compact value + label display: "12 Steps" with themed colors

CREATE  src/web/components/shared/empty_state.py
  "Nothing to show yet" placeholder with icon
```

#### Step 2.3: Create SplitWorkspaceLayout
```
CREATE  src/web/layouts/__init__.py
CREATE  src/web/layouts/base.py
  BaseLayout protocol: build(page, view_model, session) → ft.Control

CREATE  src/web/layouts/split_workspace.py
  SplitWorkspaceLayout.build(page, view_model, session) → ft.Column:
    TOP ROW (expand=3): ft.Row with 3 zones
      LEFT (expand=1): ft.Container with environment_zone_bg tint + surface_border
      CENTER (expand=2): ft.Container with interaction_zone_bg
      RIGHT (expand=1): ft.Container with agent_zone_bg tint
    BOTTOM ROW: collapsible ft.Container with charts zone
    Uses spacing constants, border colors from ColorTokens
```

#### Step 2.4: Create AppShell
```
CREATE  src/web/app.py
  AppShell class:
    __init__(page): stores page ref, applies initial theme
    build_nav_rail(): reads NAV_DESTINATIONS from constants, builds ft.NavigationRail
    build_top_bar(): AppBar with theme toggle, world/policy/speed selectors
    render_view(route, session): delegates to SplitWorkspaceLayout
    on_route_change(route): handles navigation
```

#### Step 2.5: Create theme toggle component
```
CREATE  src/web/components/controls/__init__.py
CREATE  src/web/components/controls/theme_toggle.py
  ThemeToggle.build(page): ft.IconButton with sun/moon icon
  on_click: ThemeManager.toggle(page) → page.update()
```

#### Step 2.6: Wire into main.py
```
UPDATE  src/web/main.py
  main(page) now:
    theme_manager = ThemeManager()
    theme_manager.apply_theme(page, "light")
    app_shell = AppShell(page)
    page.on_route_change = app_shell.on_route_change
    page.views.append(app_shell.render_view("/", session))
    page.update()
```

#### Step 2.7: Tests
```
CREATE  tests/web/test_theme/test_tokens.py
  - Test all ColorTokens fields present (no missing keys)
  - Test LIGHT_TOKENS and DARK_TOKENS have same field set
  - Test WCAG contrast ratios ≥ 4.5:1 for text_primary on bg_primary (both modes)

CREATE  tests/web/test_theme/test_theme_manager.py
  - Test apply_theme stores tokens in page.session
  - Test get_tokens retrieves correct tokens
  - Test toggle switches mode

CREATE  tests/web/test_layouts/test_split_workspace.py
  - Test layout renders 3 zones + charts zone
  - Test zone expand ratios (1, 2, 1)
  - Test zone background colors match tokens
  - Test charts zone collapsible behavior

CREATE  tests/web/test_components/test_shared.py
  - Test section_header renders with correct accent color
  - Test metric_badge displays value + label
  - Test empty_state renders placeholder text
```

**Commit:** `feat(web): add theme system, AppShell, SplitWorkspaceLayout, and shared components`

---

### Phase 3: Environment & Agent Zone Components

**Goal:** Left (environment) and right (agent) zones render real data from the simulator.

#### Step 3.1: Environment zone components
```
CREATE  src/web/components/environment/__init__.py

CREATE  src/web/components/environment/world_card.py
  WorldCard.build(page, config: WorldConfig):
    ft.Container with world illustration icon
    ft.Text(config.title, size=HEADING)
    ft.Text(config.description, size=BODY)
    Accent: environment_accent color strip

CREATE  src/web/components/environment/context_display.py
  ContextDisplay.build(page, context: dict, feature_order: list[str]):
    For each feature: ft.Row with key label + value badge
    Each badge uses teal-tinted background
    Pulse animation on value change (scale 100%→105%→100%, 300ms)

CREATE  src/web/components/environment/hidden_truth_panel.py
  HiddenTruthPanel.build(page, arm_defs, config: WorldConfig):
    Collapsible ft.ExpansionTile with 🔒 icon when collapsed
    When expanded: ft.Slider per arm for base_rate (0.0–1.0, step=0.01)
    Slider label shows current value as percentage
    on_change: updates world config's arm base_rates → rebuilds simulator
```

#### Step 3.2: Agent zone components
```
CREATE  src/web/components/agent/__init__.py

CREATE  src/web/components/agent/knowledge_table.py
  KnowledgeTable.build(page, arms, pull_counts, mean_rewards):
    ft.DataTable with columns: Arm, Pulls, Mean Reward
    Amber accent key line on each row
    Cells flash amber for 400ms on value change (via animation timer)
    Mean rewards displayed with "~" prefix: "~0.72"

CREATE  src/web/components/agent/pull_counter.py
  PullCounter.build(page, arm_labels, pull_counts):
    Horizontal ft.Row of compact ft.Container bars
    Each bar width proportional to pull count
    Bar fill uses agent_accent color with opacity gradient
    Labels below bars

CREATE  src/web/components/agent/uncertainty_display.py
  UncertaintyDisplay.build(page, bounds_per_arm):
    For UCB policies: horizontal bar per arm showing selected (marker) between lower/upper bounds
    ft.Container with gradient fill from left (certain) to right (uncertain)
    Only visible for UCB-family policies

CREATE  src/web/components/agent/policy_state_card.py
  PolicyStateCard.build(page, policy_id, policy_instance):
    Reads policy metadata: epsilon, alpha, lambda, etc.
    ft.Card with key-value rows
    Example: "Epsilon: 0.1 | Exploration: 10%"
```

#### Step 3.3: Wire into SplitWorkspaceLayout
```
UPDATE  src/web/layouts/split_workspace.py
  LEFT zone now calls:
    WorldCard.build(page, config)
    ContextDisplay.build(page, context, feature_order)
    HiddenTruthPanel.build(page, arm_defs, config) [if lesson allows]
  RIGHT zone now calls:
    KnowledgeTable.build(page, arms, pulls, means)
    PullCounter.build(page, labels, counts)
    PolicyStateCard.build(page, policy_id, policy)
    UncertaintyDisplay.build(page, bounds) [if UCB family]
```

#### Step 3.4: Extract selectors from main.py
```
CREATE  src/web/components/controls/world_selector.py
  WorldSelector.build(page, current_world_id, on_change):
    ft.Dropdown with all world IDs from registry
    Width: 200px, styled with theme tokens

CREATE  src/web/components/controls/policy_selector.py
  PolicySelector.build(page, current_policy_id, on_change):
    ft.Dropdown using POLICY_LABELS from policy_capabilities
    Width: 240px, styled with theme tokens

CREATE  src/web/components/controls/speed_slider.py
  SpeedSelector.build(page, current_speed, on_change):
    ft.Dropdown with ["0.25x", "0.5x", "1x", "2x", "4x", "8x"]
    Width: 100px
```

#### Step 3.5: State management
```
CREATE  src/web/state/__init__.py

CREATE  src/web/state/event_bus.py
  EventBus class:
    _subscribers: dict[str, list[Callable]]
    subscribe(event_name, callback)
    emit(event_name, **data): calls all subscribers
    Event names: STEP_COMPLETED, ARM_SELECTED, REWARD_RECEIVED, KNOWLEDGE_UPDATED,
                 THEME_CHANGED, WORLD_CHANGED, POLICY_CHANGED, RESET_TRIGGERED

CREATE  src/web/state/app_state.py
  AppState dataclass (not frozen — this is the mutable runtime state):
    theme_mode: str ("light" | "dark")
    route: str
    simulator: DiscreteSimulator | None
    preferences: UserPreferences
    lesson_progress: LessonProgressState | None
    lesson_config: LessonConfig | None
    event_bus: EventBus
    autoplay_speed: float
    current_interaction_phase: InteractionPhase

CREATE  src/web/state/interaction_state.py
  InteractionPhase enum: IDLE, CONTEXT_GENERATED, ARM_SELECTED, REWARD_RECEIVED, KNOWLEDGE_UPDATED

CREATE  src/web/state/simulation_controller.py
  SimulationController class:
    step_once(): runs 1 simulator step, advances through 4 phases with asyncio.sleep, emits events
    run_n(n): runs n steps without animation, emits events for final state
    reset(): resets simulator, emits RESET_TRIGGERED
    change_world(world_id): rebuilds simulator, emits WORLD_CHANGED
    change_policy(policy_id): rebuilds simulator, emits POLICY_CHANGED
    set_speed(multiplier): updates speed
```

#### Step 3.6: Tests
```
CREATE  tests/web/test_state/test_event_bus.py
  - Test subscribe and emit delivers events
  - Test multiple subscribers all receive event
  - Test unsubscribe removes callback
  - Test emit with no subscribers doesn't error
  - Test error in one subscriber doesn't block others

CREATE  tests/web/test_state/test_app_state.py
  - Test initial state values
  - Test state mutation through controller

CREATE  tests/web/test_state/test_simulation_controller.py
  - Test step_once advances simulator by 1
  - Test step_once emits all 4 events in order
  - Test step_once animation delays respect speed multiplier
  - Test run_n advances n steps
  - Test run_n skips animation delays
  - Test reset clears simulator state
  - Test change_world rebuilds with new world
  - Test change_policy rebuilds with new policy

CREATE  tests/web/test_components/test_environment.py
  - Test world_card renders world name + description
  - Test context_display shows all features
  - Test context_display updates on new context values
  - Test hidden_truth_panel sliders range 0.0–1.0
  - Test hidden_truth_panel slider change rebuilds simulator

CREATE  tests/web/test_components/test_agent.py
  - Test knowledge_table shows all arms
  - Test knowledge_table mean rewards have ~ prefix
  - Test knowledge_table updates on new data
  - Test pull_counter bars proportional to counts
  - Test policy_state_card shows epsilon/alpha for relevant policies
  - Test uncertainty_display only visible for UCB family
```

**Commit:** `feat(web): add environment/agent zone components, selectors, state management with event bus`

---

### Phase 4: Interaction Loop & Charts

**Goal:** Center zone shows 4-phase animated interaction. Bottom zone renders real-time animated Flet charts.

#### Step 4.1: Interaction zone components
```
CREATE  src/web/components/interaction/__init__.py

CREATE  src/web/components/interaction/step_indicator.py
  StepIndicator.build(page, current_step, horizon):
    ft.ProgressBar(value=current_step/horizon) with label: "Step 12/100"
    Themed colors

CREATE  src/web/components/interaction/arm_cards.py
  ArmCards.build(page, arms, selected_arm_id):
    ft.Row of ft.Container cards per arm
    Each card: arm label, predicted score (from policy), selection state
    Selected card: agent_accent border (amber), scale animation 105%→100%
    Unselected cards: muted, smaller
    Hover: slight elevation change

CREATE  src/web/components/interaction/reward_feedback.py
  RewardFeedback.build(page):
    Overlay component (ft.Stack with positioned child)
    Shows on REWARD_RECEIVED event:
      Success (reward=1): Green circular pulse expanding from center, fades in 600ms
      Failure (reward=0): Red shake animation on arm card area, 600ms
    Auto-hides after animation

CREATE  src/web/components/interaction/loop_visualizer.py
  LoopVisualizer.build(page, phase: InteractionPhase):
    Visualizes which of 4 phases is active
    4 numbered circles in a row: ① → ② → ③ → ④
    Active circle: filled with accent color, pulsing
    Completed circles: solid filled, neutral color
    Pending circles: outline only
    Directional arrows between circles colored:
      → (teal): Environment → Agent data flow
      ← (amber): Agent → Environment selection
```

#### Step 4.2: Chart components
```
CREATE  src/web/components/charts/__init__.py
CREATE  src/web/components/charts/chart_theme.py
  build_chart_style(tokens: ColorTokens) → dict:
    grid_color, text_color, line_colors, bar_fill colors, transparent background

CREATE  src/web/components/charts/regret_chart.py
  RegretChart.build(page, points: list[tuple[int, float]]):
    ft.LineChart with single series
    curved=True, stroke_width=2
    Animated: animated=True, animation_duration=300
    Tooltip: shows (step, regret) on hover
    Colors from chart_theme
    Update method: append point, trim to last 100

CREATE  src/web/components/charts/arm_histogram.py
  ArmHistogram.build(page, pull_counts: dict[str, int]):
    ft.BarChart with one BarChartGroup
    Bars colored with agent_accent
    animated=True, animation_duration=300
    Tooltip per bar: "Priority: 12 pulls"
    Y-axis: integer steps
    Update method: rebuild bar_rosters with new counts

CREATE  src/web/components/charts/reward_timeline.py
  RewardTimeline.build(page, steps: list[float]):
    Compact sparkline (small ft.LineChart)
    Shows last 20 reward values as step function
    Color: success_feedback (green) for 1.0, regret_feedback (red) for 0.0
    Animated: True

CREATE  src/web/components/charts/knowledge_heatmap.py
  KnowledgeHeatmap.build(page, arm_buckets, context_bins):
    Only for contextual bandits (LinUCB, LinTS, etc.)
    Grid of ft.Container cells
    Rows = arms, Columns = context bins
    Cell color intensity = estimated mean reward (white→amber gradient)
    Cell tooltip: "Arm X, Context bin Y: ~0.72"
    Update: rebuild grid with new estimates
```

#### Step 4.3: Step controls
```
CREATE  src/web/components/controls/step_controls.py
  StepControls.build(page, controller: SimulationController):
    ft.Row:
      ft.ElevatedButton("Step ▸", on_click=controller.step_once)
      ft.ElevatedButton("▶ Play" or "⏸ Pause", on_click=toggle_play)
      ft.ElevatedButton("↺ Reset", on_click=controller.reset)
      ft.TextField("50", width=60) + ft.ElevatedButton("Run N", on_click=controller.run_n)
    Play/Pause button text changes based on state
    All buttons use themed colors
    Step button also shows keyboard shortcut "S"
```

#### Step 4.4: Wire into SplitWorkspaceLayout
```
UPDATE  src/web/layouts/split_workspace.py
  CENTER zone now calls:
    LoopVisualizer.build(page, phase) — subscribed to phase changes
    StepIndicator.build(page, step, horizon)
    ArmCards.build(page, arms, selected) — subscribed to ARM_SELECTED
    RewardFeedback.build(page) — overlaid, shown on REWARD_RECEIVED
    StepControls.build(page, controller)

  BOTTOM zone (collapsible ft.ExpansionTile):
    ft.Row:
      RegretChart.build(page, points)
      ArmHistogram.build(page, counts)
    ft.Row (for contextual bandits):
      RewardTimeline.build(page, rewards)
      KnowledgeHeatmap.build(page, arm_buckets, context_bins)
```

#### Step 4.5: Refactor main.py autoplay into controller
```
UPDATE  src/web/main.py
  Remove _autoplay_loop, _on_play, _on_step, _on_reset, _navigate_to
  Replace with SimulationController methods
  _on_world_change + _on_policy_change → merge into controller.change_world/change_policy

  main.py becomes ~50 lines: just ft.app(target=app.main) + run()
```

#### Step 4.6: Tests
```
CREATE  tests/web/test_components/test_interaction.py
  - Test step_indicator shows correct progress fraction
  - Test arm_cards highlights selected arm with amber border
  - Test reward_feedback shows green for reward=1, red for reward=0
  - Test reward_feedback auto-hides after 600ms
  - Test loop_visualizer shows correct phase highlights
  - Test loop_visualizer arrow directions correct per phase

CREATE  tests/web/test_components/test_charts.py
  - Test regret_chart builds with correct data points
  - Test regret_chart limits to 100 points
  - Test arm_histogram bars proportional to pull counts
  - Test arm_histogram tooltips show arm name + count
  - Test reward_timeline colors: green for 1.0, red for 0.0
  - Test knowledge_heatmap only renders for contextual policies
  - Test knowledge_heatmap cell colors correspond to mean rewards

CREATE  tests/web/test_components/test_controls.py
  - Test step button calls controller.step_once
  - Test play/pause toggles text and calls controller
  - Test reset button calls controller.reset
  - Test Run-N input validates positive integer
  - Test Run-N button calls controller.run_n with input value
  - Test speed selector updates speed multiplier

CREATE  tests/web/test_e2e.py (UPDATE existing)
  - Test complete step cycle: click Step → 4 phases animate → charts update
  - Test autoplay: click Play → multiple steps run → charts accumulate
  - Test Run-50: click Run N → 50 steps complete → charts show 50 points
  - Test reset: run steps → click Reset → all state cleared
  - Test world change: select new world → simulator rebuilt → context changes
  - Test policy change: select new policy → simulator rebuilt → knowledge resets
```

**Commit:** `feat(web): add animated interaction loop, Flet native charts, and step controls`

---

### Phase 5: Theme Integration & Visual Polish

**Goal:** Every component reads from ColorTokens. Dark mode toggle works globally. No hardcoded hex values remain.

#### Step 5.1: Audit and replace all hardcoded styles
```
AUDIT every .py file in src/web/ for hardcoded color hex codes, font sizes, spacing values
  Replace "#D5D7DA" → tokens.surface_border
  Replace "#D84315" → tokens.regret_feedback
  Replace hardcoded sizes (14, 12, 11, 10, 9) → FontScale constants
  Replace hardcoded padding (12, 20) → SpacingScale constants
  Replace "COBA · " prefix → constant COBA_BRANDING
  Replace magic number 5 → MAX_LESSON_STAGES constant

Ensure EVERY component calls ThemeManager.get_tokens(page) for styling
```

#### Step 5.2: Animation constants
```
UPDATE all animation durations to use AnimationDurations from theme/constants.py
  phase 1: ANIM_PHASE_CONTEXT (300ms)
  phase 2: ANIM_PHASE_ARM (400ms)
  phase 3: ANIM_PHASE_REWARD (600ms)
  phase 4: ANIM_PHASE_KNOWLEDGE (300ms)
  charts: ANIM_CHART (300ms)
```

#### Step 5.3: Navigation rail styling
```
UPDATE  src/web/app.py build_nav_rail():
  Read colors from tokens
  Use ft.NavigationRailDestination with themed icons
  Active indicator uses environment_accent
```

#### Step 5.4: Tests
```
CREATE  tests/web/test_theme/test_integration.py
  - Test dark mode toggle changes ALL zone backgrounds
  - Test dark mode toggle changes chart colors
  - Test dark mode toggle changes text colors throughout
  - Test dark mode persists across page navigation

CREATE  tests/web/test_ui_smoke.py (UPDATE existing)
  - Add assertions that rendered widgets use tokens, not hardcoded hex
  - Test all 5 routes render without exception in both themes
```

**Commit:** `feat(web): integrate theme system across all components, remove hardcoded styles`

---

### Phase 6: Curriculum & Sandbox Routes

**Goal:** Lesson mode works with narrative. Sandbox mode has full parameter editing. Comparison mode works.

#### Step 6.1: Lesson mode enhancements
```
UPDATE  src/web/curriculum/lessons.py
  Add narrative_banner field to LessonConfig: "You are a doctor at a rural clinic..."
  Add illustration_id field for world-specific SVG

UPDATE  src/web/layouts/split_workspace.py (Lesson variant)
  Add NarrativeBanner component above the 3 zones
  ft.Container with illustration + narrative text, teal accent left border
  HiddenTruthPanel: locked (🔒) until Stage 3+, collapsed by default
  LessonObjectiveCard: overlay at bottom showing current objective + progress

CREATE  src/web/components/shared/lesson_progress_bar.py
  Horizontal dot indicators: ● ○ ○ ○ ○ (filled for completed, outlined for remaining)
  "Stage 2 of 5" label
```

#### Step 6.2: Sandbox mode
```
UPDATE  src/web/layouts/split_workspace.py (Sandbox variant)
  HiddenTruthPanel: expanded by default, no lock icon
  "Generate Custom World" button:
    Opens ft.AlertDialog with form: world name, features (add/remove), arms (add/remove)
    Builds WorldConfig from form → rebuilds simulator
  Real-time preview: when probability slider changes, KnowledgeTable updates expected values
```

#### Step 6.3: Home route
```
UPDATE  src/web/layouts/split_workspace.py (Home variant)
  Hero area: large COBA title + "Learn how machines make decisions under uncertainty"
  Two large navigation cards with icons:
    "Start Learning" → /lesson (narrative path)
    "Free Play" → /arena (unguided)
  Feature highlights row: "17 Algorithms · 7 Real Worlds · Real-Time Charts"
```

#### Step 6.4: Comparison mode
```
UPDATE  src/web/layouts/split_workspace.py (Comparison variant)
  Top: multi-select ft.Dropdown for 2-3 policies (multi-select via checkboxes)
  Bottom zone with 3 charts:
    RegretOverlay: multiple colored lines on same chart + legend
    ArmDistributionComparison: grouped ft.BarChart (side-by-side bars)
    SummaryMetricsTable: ft.DataTable with cumulative reward, final regret, best-arm rate
  "Run Comparison (100 steps)" button
```

#### Step 6.5: Tests
```
CREATE  tests/web/test_e2e_lesson.py
  - Test lesson stage auto-advances when objective met
  - Test HiddenTruthPanel is locked in early stages
  - Test narrative banner changes per lesson
  - Test lesson progress bar shows correct stage

CREATE  tests/web/test_e2e_sandbox.py
  - Test probability slider changes arm base_rate
  - Test agent adapts after multiple steps with new probabilities
  - Test "Generate Custom World" creates valid WorldConfig

CREATE  tests/web/test_e2e_comparison.py
  - Test two policies run comparison and produce different results
  - Test regret overlay chart has correct number of series
  - Test summary table shows correct metrics

CREATE  tests/web/test_e2e_home.py
  - Test home route renders two navigation cards
  - Test "Start Learning" navigates to /lesson
  - Test "Free Play" navigates to /arena
```

**Commit:** `feat(web): add lesson narrative, sandbox editing, comparison mode, and home screen`

---

### Phase 7: Edge Cases, Error Handling & Accessibility

**Goal:** Handle all error states gracefully. Keyboard navigation works. Screen reader labels present.

#### Step 7.1: Edge cases
```
Handle:
  - Empty trace (no steps run yet): EmptyState in charts zone
  - World with 2 arms vs 10 arms: layout doesn't break
  - Horizon=1 (single step): progress bar shows 100%
  - Very long arm names: truncated with ellipsis + tooltip
  - Policy that doesn't support uncertainty: UncertaintyDisplay hidden
  - Non-contextual policy: ContextDisplay shows "No context features"
  - Policy with zero pulls on some arms: mean shows "—"
  - Disconnect/reconnect: state restored from preferences
  - Rapid clicking: buttons disabled during step animation
  - Run-N with 10000: runs in background, doesn't block UI (use page.run_task)
```

#### Step 7.2: Loading states
```
Add loading skeletons:
  - Simulator rebuild after world/policy change: skeleton in all zones
  - Run-N in progress: progress indicator in charts zone
  - Initial app load: skeleton in all zones until first render
```

#### Step 7.3: Accessibility
```
All interactive elements: tooltip or aria-label
All charts: alt text describing what they show
Keyboard shortcuts:
  S: Step
  Space: Play/Pause
  R: Reset
  ← →: Navigate between routes
  D: Toggle dark mode
```

#### Step 7.4: Tests
```
CREATE  tests/web/test_edge_cases.py
  - Test empty trace shows empty state in charts
  - Test single-arm world renders without error
  - Test 20-arm world renders without overflow
  - Test long arm names truncated with tooltip
  - Test rapid clicking doesn't cause state corruption
  - Test disconnect/reconnect preserves state
  - Test Run-N with 10000 doesn't block UI
  - Test world change mid-autoplay cancels and rebuilds
```

**Commit:** `fix(web): handle edge cases, add loading states, improve accessibility`

---

## 5. Verification & Testing Strategy

### Test Organization
```
tests/web/
├── conftest.py                          # Shared fixtures (any_world, any_policy, seed_rng, mock_page)
├── _stubs.py                            # DummyWorld, GreedyStubPolicy, etc. (keep, extend)
│
├── test_theme/                          # Phase 2
│   ├── test_tokens.py
│   ├── test_theme_manager.py
│   └── test_integration.py             # Phase 5
│
├── test_state/                          # Phase 3
│   ├── test_event_bus.py
│   ├── test_app_state.py
│   └── test_simulation_controller.py
│
├── test_layouts/                        # Phase 2
│   └── test_split_workspace.py
│
├── test_components/                     # Phases 3-4
│   ├── test_shared.py
│   ├── test_environment.py
│   ├── test_agent.py
│   ├── test_interaction.py
│   ├── test_charts.py
│   └── test_controls.py
│
├── test_e2e.py                          # Phase 4 (UPDATE existing)
├── test_e2e_lesson.py                   # Phase 6
├── test_e2e_sandbox.py                  # Phase 6
├── test_e2e_comparison.py              # Phase 6
├── test_e2e_home.py                    # Phase 6
│
└── test_edge_cases.py                   # Phase 7
```

### Existing tests to UPDATE (not delete)
```
tests/web/test_contracts.py       → imports may change if models.py path changes
tests/web/test_state.py           → RunConfig/SimulationState still exist
tests/web/test_simulator.py       → DiscreteSimulator unchanged
tests/web/test_worlds.py          → WorldConfig still same
tests/web/test_world_schema.py    → Schema unchanged
tests/web/test_router.py          → Route specs unchanged
tests/web/test_main.py            → Significant updates as main.py shrinks
tests/web/test_view_models.py     → RouteUIModel still same, imports update
tests/web/test_preferences.py     → PreferencesStore unchanged
tests/web/test_curriculum.py      → LessonConfig gains narrative_banner field
tests/web/test_trace.py           → TraceBuffer unchanged
tests/web/test_arena.py           → moves to test_analysis/
tests/web/test_comparison.py      → moves to test_analysis/
tests/web/test_ui_smoke.py        → UPDATE for new component structure
tests/web/test_ui_layout.py       → UPDATE for SplitWorkspaceLayout
```

### Existing tests to DELETE (testing deleted code)
```
tests/web/test_shell.py           → shell.py is deleted
```

### Run commands
```bash
# After each phase:
uv run pytest tests/web/ -v -p no:asyncio

# Full suite:
uv run pytest tests/ -v -p no:asyncio --ignore=tests/test_shared_sim.py

# With coverage:
uv run pytest tests/web/ -v -p no:asyncio --cov=src/web --cov-report=term-missing
```

---

## 6. Summary of All File Operations

### Commands to run after each phase:
```bash
# Phase 1
uv run pytest tests/web/ -v -p no:asyncio    # Must still pass
# Commit: refactor(web): delete dead code, consolidate dataclasses, extract policy bases

# Phase 2
uv run pytest tests/web/ -v -p no:asyncio    # + new theme/layout tests
# Commit: feat(web): add theme system, AppShell, SplitWorkspaceLayout

# Phase 3
uv run pytest tests/web/ -v -p no:asyncio    # + new component/state tests
# Commit: feat(web): add environment/agent zone components and state management

# Phase 4
uv run pytest tests/web/ -v -p no:asyncio    # + new interaction/chart tests
# Commit: feat(web): add animated interaction loop and Flet native charts

# Phase 5
uv run pytest tests/web/ -v -p no:asyncio    # + theme integration tests
# Commit: feat(web): integrate theme system, remove hardcoded styles

# Phase 6
uv run pytest tests/web/ -v -p no:asyncio    # + new E2E tests
# Commit: feat(web): add lesson narrative, sandbox, comparison, home routes

# Phase 7
uv run pytest tests/web/ -v -p no:asyncio    # + edge case tests
u# Commit: fix(web): handle edge cases, add loading states, improve accessibility
```
