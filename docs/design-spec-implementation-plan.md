# COBA UI Design Spec — Comprehensive Implementation Plan

> **Source:** `COBA_UI_DESIGN_SPEC.md` v3.0
> **Date:** 2025-05-25
> **Current state:** Flet 0.85.1 codebase with theme system, SplitWorkspaceLayout, 775 passing tests

---

## 0. Context: What Exists vs. What's Needed

### What's Already Done
- `theme/` — ColorTokens, ThemeManager, SpacingScale, FontScale, AnimationDurations
- `layouts/` — SplitWorkspaceLayout (3-zone + bottom charts ExpansionTile)
- `components/` — environment.py, agent.py, interaction.py, charts.py, shared.py, theme_toggle.py
- `statemgmt/` — EventBus, InteractionPhase
- `app.py` — AppShell with NavigationRail, routing, view rendering
- `main.py` — _SimSession, autoplay loop, pref persistence

### What Needs Work (Spec §9)
| # | Fix | Priority |
|---|---|---|
| 1 | Wire autoplay with `asyncio.create_task` | P0 |
| 2 | Connect chart data to Flet rendering | P0 |
| 3 | Wire lesson `advance()` from UI | P0 |
| 4 | Use `build_arm_cards()` in `app.py` | P1 |
| 5 | Re-render after theme toggle | P1 |
| 6 | Connect SandboxEditor to Sandbox route | P1 |
| 7 | Connect debug panes to Arena/Agent | P2 |
| 8 | Fix global session state for multi-user | P0 |
| 9 | Respect layout width ratios | P1 |
| 10 | Implement Comparison route | P2 |
| 11 | Min-width button labels (no space padding) | P1 |
| 12 | Responsive pull bars (not 100px fixed) | P1 |
| 13 | CAPTION=11 (not 10) | P0 |
| 14 | Neutral reward feedback for 0.0 | P1 |

---

## Phase 1: Bug Fixes & Foundation (P0 items)

**Goal:** Fix critical bugs. Update tokens. No visual redesign yet.

### Step 1.1: Fix `CAPTION = 11` (from current `10`)
Spec §2.2 requires 11px minimum for WCAG legibility.
```
UPDATE src/web/theme/constants.py: FontScale.CAPTION = 11
UPDATE all components using CAPTION — they'll inherit the new size
```

### Step 1.2: Fix module-level globals for multi-user web
Spec §9.8 — `main.py` uses `_page`, `_session`, `_pref_store` as module globals. For Flet web mode, each user gets their own `main(page)` call. Move these into `page.data` to prevent session leaks between tabs.

```
UPDATE src/web/main.py:
  Store _session in page.data["_session"] instead of module global
  Store _pref_store in page.data["_pref_store"] instead of module global
  Store _shell in page.data["_shell"] instead of module global
  Remove module-level global variables
UPDATE src/web/app.py:
  AppShell no longer takes session/pref_store in __init__ — reads from page.data
```

### Step 1.3: Re-render after theme toggle
Spec §9.5 — theme toggle changes `page.theme_mode` but never calls `_refresh_view()`.
```
UPDATE src/web/components/theme_toggle.py:
  In _on_toggle, call ThemeManager.toggle(page), then trigger full refresh
UPDATE src/web/app.py:
  Expose _refresh_view publicly or use EventBus THEME_CHANGED event
```

### Step 1.4: Wire autoplay with `asyncio.create_task`
Spec §9.1 — autoplay loop exists in `main.py` but `AppShell._autoplay_task` is never started. The `_SimSession.do_play()` method exists, but the AppShell doesn't use it.

```
UPDATE src/web/app.py:
  Restore _autoplay_loop from original main.py
  Wire play/pause button callbacks to start/cancel autoplay
  Step/Pause/Reset callbacks from _SimSession
UPDATE src/web/main.py:
  Wire AppShell callbacks to _SimSession methods
```

### Step 1.5: Wire lesson `advance()` from UI
Spec §9.3 — `LessonProgressState.advance()` exists but is never called from any UI button. The original `main.py` had `_advance_lesson_if_ready` which auto-advanced, but the new AppShell doesn't call it.

```
UPDATE src/web/app.py:
  In _render_view → path for lesson route:
    Add "Next Stage →" button when objective is met
    Wire button callback to session.lesson_progress.advance()
    Call _refresh_view() after advancement
```

### Tests for Phase 1
```
CREATE tests/web/test_phase1_foundation.py:
  - test_caption_is_11: FontScale.CAPTION == 11
  - test_session_is_per_page_not_global: page.data has isolated session
  - test_theme_toggle_calls_refresh_view: mock AppShell._refresh_view, verify called
  - test_autoplay_loop_starts_on_play: mock asyncio, verify task created
  - test_autoplay_loop_cancels_on_pause: verify task canceled
  - test_lesson_advance_button_when_objective_met: set objective state → button renders
  - test_lesson_advance_button_not_shown_when_objective_unmet: button absent
```

**Commit:** `fix(web): P0 fixes — CAPTION=11, per-page session, theme refresh, autoplay, lesson advance`

---

## Phase 2: Layout & Navigation Redesign (Spec §§3-5, P1)

**Goal:** Replace NavigationRail with top nav. Home page redesign. Unified workspace surface.

### Step 2.1: Top Navigation Bar (replace NavigationRail + sidebar)
Spec §3.1 — compact top nav of 46px height, full-width. Logo + nav tabs + world/policy selectors.

```
CREATE src/web/components/top_nav.py:
  build_top_nav(page, current_route, world_id, policy_id, on_navigate, on_world_change, on_policy_change):
    ft.Container(height=46, bgcolor=bg_primary, border=bottom 0.5px surface_border)
      left: "COBA" 14px/500 + "Bandit Lab" 11px text_secondary
      center: nav tabs (Home|Lesson|Arena|Sandbox|Compare) — icon-only for Home, text for rest
      right: world select <select> + policy select <select> + theme toggle

DELETE src/web/app.py: _build_nav_rail() method (replaced by top nav)
UPDATE src/web/app.py: _build_top_bar → _build_top_nav (no longer returns ft.AppBar, returns Row)
UPDATE src/web/app.py: _render_view → remove ft.AppBar, add top nav at top of content column
```

### Step 2.2: Unified Workspace Surface (no individual zone cards)
Spec §0.2 — single bordered container, not three separate cards. Internal column dividers via `ft.VerticalDivider`.

```
UPDATE src/web/layouts/split_workspace.py:
  Single ft.Container with border_radius=10, border=0.5px surface_border
  Inside: ft.Row with 3 zones separated by VerticalDivider
  Each zone: ft.Column with padding=11_12, no individual border, no bgcolor tint
  Zone header: 6px colored dot + UPPERCASE title (10px/500/text_secondary), accent=2px top border on zone
UPDATE src/web/theme/tokens.py:
  Rename token values per spec §2.1. Remove environment_zone_bg/agent_zone_bg tints.
  Add border_tertiary for 0.5px internal dividers.
  Consolidate to single accent #059669 (drop amber).
```

### Step 2.3: Home Page Redesign
Spec §4 — hero + destination cards + continue card.

```
UPDATE src/web/app.py: _render_view for HOME route:
  Hero: "Contextual Bandit Lab" 20px/500 + subtitle 13px/text_secondary
  Destination cards (2x2 grid):
    Lesson (school icon, environment_accent bg), Arena (chart-line, blue),
    Sandbox (flask, amber), Compare (columns, gray)
    Each card: white bg, border_radius=12, icon+title+desc+"Open →" link
  Continue card: if session has lesson_progress, show resume card
```

### Step 2.4: Color Token Updates
Spec §2.1 — update all token values to match the spec. Single teal accent `#059669`. Remove amber from agent zone.

```
UPDATE src/web/theme/tokens.py:
  Replace all light/dark token values with spec's color table
  environment_zone_bg, agent_zone_bg → use bg_primary (unified surface)
  agent_accent → remove, use neutral or teal everywhere
  chart_line_reward, chart_line_regret → spec values
  text_muted → spec values
  ADD border_tertiary token (0.5px internal dividers)
```

### Tests for Phase 2
```
CREATE tests/web/test_phase2_layout.py:
  - test_top_nav_renders_five_tabs
  - test_top_nav_active_tab_highlighted
  - test_workspace_is_single_container_not_three_cards
  - test_workspace_has_vertical_dividers
  - test_home_renders_hero_section
  - test_home_renders_four_destination_cards
  - test_home_destination_cards_have_navigation_links
  - test_color_tokens_have_border_tertiary
  - test_color_tokens_single_accent_no_amber
  - test_zone_headers_are_uppercase_10px_text_secondary
```

**Commit:** `feat(web): P1 layout — top nav, unified surface, home page, single teal accent`

---

## Phase 3: Component Upgrades (Spec §6, P1)

**Goal:** Arm cards with score bars, hover-reveal, responsive pull bars, neutral reward feedback.

### Step 3.1: Arm Cards Redesign
Spec §6.2 — from colored capsules to rows with left-border accent, hover-reveal scores.

```
UPDATE src/web/components/interaction.py: build_arm_cards():
  Instead of colored ft.Container with bg, use:
  - ft.Row for each arm: label (13px/500) + score (12px/mono, hidden until hover) + score bar
  - Selected arm: border-left 3px environment_accent + subtle bg tint
  - Hover: border all sides 0.5px environment_accent
  - Score bar: 3px height, proportional to predicted_score
DELETE arm card bgcolor/selected_glow logic (old amber system)
```

### Step 3.2: Responsive Pull Bars
Spec §6.11 — use proportion-based width, not fixed 100px.

```
UPDATE src/web/components/agent.py: build_pull_counter():
  Replace width=int(100 * pct) with ft.Row using expand ratios
  Track: ft.Container(expand=count_vs_max) + ft.Container(expand=max-count_vs_max)
  Labels: 11px, truncate with ellipsis
  Best arm: environment_accent fill, others: text_muted
```

### Step 3.3: Neutral Reward Feedback
Spec §6.13 — reward=0.0 is not failure. Use neutral styling.

```
UPDATE src/web/components/interaction.py: build_reward_feedback():
  Reward > 0: green "↑ reward" + numeric delta
  Reward = 0: "No reward" in text_muted (not red)
  Reward < 0: red "↓ penalty"
  Show cumulative reward as subtext
  Use animate_opacity for 600ms fade-out
```

### Step 3.4: Hover-Reveal Knowledge Table Values
Spec §0.4 — computed values (means, scores) hidden until hover.

```
UPDATE src/web/components/agent.py: build_knowledge_table():
  Mean values: color=tokens.text_primary (always visible — spec says counts+labels always visible)
  Applies to both arm name (always visible) and mean reward (always visible)
  Hover: entire row gets subtle bg tint
```

### Step 3.5: Inline Formula and Tuning Hints Below Sliders
Spec §6.8 — ParamTooltip content always visible, not behind hover.

```
UPDATE src/web/components/agent.py: build_policy_state_card():
  For each parameter, show inline formula (11px/mono/text_secondary) + tuning hint (11px/italic/text_muted)
UPDATE src/web/ui/param_controls.py:
  ParamTooltip already has formula and tuning_hint fields — ensure they're surfaced in the UI
```

### Step 3.6: Min-Width Button Labels
Spec §6.4 — no space-padded labels, use `min_width` prop.

```
UPDATE src/web/components/interaction.py: build_step_controls():
  Replace "  Step (42)  " → text="Step", badge=f"({current_step})" beside button
  Replace "  ▶ Play  " → text="Play"/"Pause" (based on state) + icon
  Replace "  ↺ Reset  " → text="Reset" + icon
  Add min_width=80 to each button
```

### Tests for Phase 3
```
CREATE tests/web/test_phase3_components.py:
  - test_arm_cards_have_score_bar
  - test_arm_cards_selected_has_left_border
  - test_arm_cards_score_hidden_until_hover
  - test_pull_bars_use_proportional_width
  - test_pull_bars_best_arm_is_accent_colored
  - test_reward_feedback_neutral_for_zero
  - test_reward_feedback_positive_shows_delta
  - test_reward_feedback_negative_is_red
  - test_step_controls_no_space_padding
  - test_step_controls_buttons_have_min_width
  - test_param_slider_shows_formula_inline
```

**Commit:** `feat(web): P1 components — arm cards, pull bars, neutral feedback, inline params`

---

## Phase 4: Lesson & Theory Enhancements (P1-P2)

**Goal:** Stage stepper, theory card, objective meters, lesson advance button.

### Step 4.1: Stage Stepper Above Workspace
Spec §6.8 — horizontal dots + connectors, labels on hover.

```
CREATE src/web/components/stepper.py:
  build_stage_stepper(page, current_stage, total_stages):
    ft.Row of dots (8px circles) connected by 0.5px lines
    Completed: environment_accent fill, pointer cursor
    Active: environment_accent fill, scale 1.4
    Upcoming: bg_secondary fill, surface_border border
    Labels: 10px below, transparent by default, visible on hover or active
UPDATE src/web/app.py: lesson route prepends stage stepper above workspace
```

### Step 4.2: Theory Card Below Workspace
Spec §6.7 — full-width, below 3-pane row.

```
CREATE src/web/components/theory_card.py:
  build_theory_card(page, lesson_panel: LessonPanelModel):
    ft.Container: full width, border_radius=10, border=0.5px surface_border
    Header: "STAGE X · TITLE" 11px/uppercase/environment_accent
    Stepper mini: [1/5 > 2/5 > ...] inside header
    Formula block: monospace, bg_tertiary, border=surface_border, border_radius=6
    Intuition text: 13px/text_secondary
    Hint text: 12px/italic/text_muted
    Objective meters: two progress bars (Steps + Reward)
UPDATE src/web/app.py: lesson route appends theory card below workspace
```

### Step 4.3: Objective Meters
Spec §6.9 — progress bars inside theory card.

```
UPDATE src/web/components/theory_card.py:
  build_objective_meters(objective, current_reward, current_steps):
    Steps: 42/80 bar
    Reward: 22.4/36 bar
    Track: bg_tertiary, height 6px, border_radius=3
    Fill: environment_accent (green), regret_feedback if regret limit close
```

### Step 4.4: Lesson "Next Stage →" Button
Spec §7.2 — appears when objective is met.

```
UPDATE src/web/components/theory_card.py:
  When evaluate_lesson_objective() returns True:
    Show "Next Stage →" button (teal filled)
    on_click: advance lesson progress + refresh view
UPDATE src/web/curriculum/lessons.py:
  LessonConfig already has objective and stages. No changes needed.
```

### Tests for Phase 4
```
CREATE tests/web/test_phase4_lesson.py:
  - test_stepper_shows_five_stages
  - test_stepper_active_stage_is_accent_filled
  - test_stepper_completed_stages_are_checkmarked
  - test_theory_card_shows_stage_title_and_formula
  - test_theory_card_has_objective_meters
  - test_next_stage_button_appears_when_objective_met
  - test_next_stage_button_advances_progress
  - test_next_stage_button_not_shown_when_incomplete
```

**Commit:** `feat(web): P1 lesson — stage stepper, theory card, objective meters, advance`

---

## Phase 5: Charts, Autoplay & Arena (P0-P1)

**Goal:** Connect chart data to Flet rendering. Wire autoplay fully. Add KPI row to Arena.

### Step 5.1: Chart Rendering in SplitWorkspaceLayout
Spec §9.2 — `build_chart_data()` exists but no Flet chart consumes it.

```
UPDATE src/web/ui/charts.py:
  build_flet_regret_chart(page, chart_data: ChartData) → ft.Container-based bar chart
  build_flet_arm_histogram(page, chart_data: ChartData) → responsive bar chart
UPDATE src/web/layouts/split_workspace.py:
  When arena_metrics is present in view model, pass chart_data to bottom zone
  Bottom zone: regret chart + arm histogram side-by-side
UPDATE src/web/app.py:
  Pass arena_metrics → ChartData to layout
```

### Step 5.2: KPI Metric Cards (Arena)
Spec §6.10 — four borderless metric cards above workspace.

```
CREATE src/web/components/kpi_row.py:
  build_kpi_row(page, steps, cum_reward, cum_regret, best_arm, best_pulls, best_pct):
    4 ft.Container in a Row:
    Steps: 247 → 22px/500, "STEPS" 11px/uppercase/text_secondary below
    Cum Reward: 148.3 → environment_accent
    Cum Regret: 43.7 → regret_feedback
    Best Arm: "Action Film" 14px/500 + "147 pulls (60%)" 11px/text_muted
    bg=bg_secondary, border_radius=8, padding=12
    No border — background contrast is enough
```

### Step 5.3: Wire Autoplay Fully
Spec §9.1 — autoplay loop running via `asyncio.create_task`.

```
UPDATE src/web/app.py:
  _start_autoplay(): creates asyncio.Task that loops step → refresh → sleep
  _stop_autoplay(): cancels task
  Wire on_play callback → _start_autoplay
  Wire on_pause callback → _stop_autoplay
  Wire on_reset callback → _stop_autoplay + session.do_reset()
  Step speed reads from session.prefs.speed
```

### Tests for Phase 5
```
CREATE tests/web/test_phase5_arena.py:
  - test_chart_data_is_passed_to_layout_when_arena_metrics_present
  - test_kpi_row_shows_steps_reward_regret_best_arm
  - test_kpi_row_reward_is_accent_colored
  - test_kpi_row_regret_is_red
  - test_autoplay_starts_async_task_on_play
  - test_autoplay_cancels_on_pause
  - test_autoplay_cancels_on_reset
  - test_autoplay_speed_respects_preferences
```

**Commit:** `feat(web): P0-P1 charts, KPI row, autoplay wiring for Arena`

---

## Phase 6: Sandbox, Comparison & Debug (P1-P2)

**Goal:** Connect SandboxEditor, implement Comparison route, connect debug panes.

### Step 6.1: SandboxEditor UI Connection
Spec §9.6 — `sandbox.py` exists but no Flet UI connects to it.

```
UPDATE src/web/app.py: sandbox route:
  Add "Configure World" expandable panel in Environment zone
  Arm base-rate sliders from SandboxEditor (0.0-1.0, step 0.05)
  "Apply" button calls SandboxEditor.build_world_override() → rebuild simulator
  Horizon slider
  Show "World overrides active" indicator when overrides are applied

UPDATE src/web/sandbox.py:
  Expose SandboxEditor.get_arm_defs() for UI enumeration
  Expose SandboxEditor.get_current_params() for UI state
```

### Step 6.2: Comparison Route Skeleton
Spec §9.10 — dual-policy side-by-side.

```
UPDATE src/web/app.py: comparison route:
  Two policy selector dropdowns (Policy A, Policy B)
  "Run Comparison" button → calls run_policy_comparison from analysis/
  Side-by-side display: regret overlay charts, arm distribution comparison
  Difference summary card below
UPDATE src/web/components/agent.py:
  Add build_comparison_summary() for delta metrics
```

### Step 6.3: Debug Pane Connection
Spec §9.7 — `AdvancedDebugPane` builders exist but aren't connected to any UI.

```
UPDATE src/web/app.py: agent zone Config tab:
  Add "Debug" section that calls appropriate debug_*_builder based on policy_capability
  Show for advanced policies (gp_ucb, bootstrapped_ensemble, linucb_hybrid, tree_ucb, tree_ts)
```

### Tests for Phase 6
```
CREATE tests/web/test_phase6_advanced.py:
  - test_sandbox_shows_arm_sliders_when_configure_expanded
  - test_sandbox_apply_button_rebuilds_simulator
  - test_comparison_runs_two_policies_and_shows_regret_overlay
  - test_comparison_summary_shows_delta_metrics
  - test_debug_pane_renders_for_gp_ucb
  - test_debug_pane_not_rendered_for_context_free_policies
```

**Commit:** `feat(web): P1-P2 sandbox editor, comparison route, debug pane connection`

---

## Phase 7: Polish & Accessibility (P2)

**Goal:** Edge cases, keyboard shortcuts, final theme token sync.

### Step 7.1: Keyboard Shortcuts
Spec mentions keyboard shortcuts implicitly. Add them to AppShell.

```
UPDATE src/web/app.py:
  page.on_keyboard_event handler:
    S → step
    Space → play/pause (when not typing in input)
    R → reset
    D → toggle theme
    Arrow Left/Right → previous/next route tab
```

### Step 7.2: Theme Token Audit
Ensure all component colors match the spec's final token values.

```
AUDIT every component for:
  Any remaining hardcoded hex values → replace with token
  Any remaining amber-colored elements → replace with neutral or teal
  Border width consistency (0.5px vs 1px vs 2px)
  Spacing consistency (8px grid)
UPDATE all mismatched values
```

### Step 7.3: Edge Cases
```
HANDLE:
  - Empty trace → empty state in all zones
  - Policy with no debug snapshot → hide debug tab
  - World with 2 arms vs 10 arms → layout adapts
  - Long arm names → truncate with ellipsis + tooltip
  - Rapid parameter changes → debounce simulator rebuilds
```

### Tests for Phase 7
```
CREATE tests/web/test_phase7_polish.py:
  - test_keyboard_s_shortcut_triggers_step
  - test_keyboard_space_toggles_play_pause
  - test_keyboard_r_triggers_reset
  - test_keyboard_d_toggles_theme
  - test_empty_trace_shows_empty_state_in_all_zones
  - test_long_arm_names_are_truncated
  - test_no_hardcoded_hex_colors_in_any_component
  - test_border_widths_match_spec
```

**Commit:** `fix(web): P2 polish — keyboard shortcuts, theme audit, edge case handling`

---

## File Change Summary

### Files to CREATE
| File | Phase |
|------|-------|
| `tests/web/test_phase1_foundation.py` | 1 |
| `src/web/components/top_nav.py` | 2 |
| `tests/web/test_phase2_layout.py` | 2 |
| `tests/web/test_phase3_components.py` | 3 |
| `src/web/components/stepper.py` | 4 |
| `src/web/components/theory_card.py` | 4 |
| `src/web/components/objective_meters.py` | 4 |
| `tests/web/test_phase4_lesson.py` | 4 |
| `src/web/components/kpi_row.py` | 5 |
| `tests/web/test_phase5_arena.py` | 5 |
| `tests/web/test_phase6_advanced.py` | 6 |
| `tests/web/test_phase7_polish.py` | 7 |

### Files to UPDATE
| File | Phases |
|------|--------|
| `theme/constants.py` | 1 |
| `theme/tokens.py` | $2$ |
| `components/theme_toggle.py` | 1 |
| `components/agent.py` | 3, 6 |
| `components/interaction.py` | 3 |
| `components/environment.py` | 2 (minor border updates) |
| `components/shared.py` | 2 (new section header style) |
| `components/charts.py` | 5 |
| `layouts/split_workspace.py` | 2, 5 |
| `app.py` | 1, 2, 4, 5, 6, 7 |
| `main.py` | 1 |
| `ui/charts.py` | 5 |
| `ui/view_models.py` | 5 |
| `sandbox.py` | 6 |

### Verdicts on Existing Code
| File | Decision |
|------|----------|
| Old `app.py` `_build_nav_rail()` | Replace with `top_nav.py` |
| Old `app.py` `_build_top_bar()` | Replace with `top_nav.py` |
| Old `split_workspace.py` `_build_zone()` | Replace with unified surface |
| Old `interaction.py` `build_arm_cards()` | Redesign per spec §6.2 |
| Old `interaction.py` `build_reward_feedback()` | Neutral-feedback redesign |
| Old `agent.py` `build_pull_counter()` | Proportional-width redesign |
| Old `agent.py` `build_policy_state_card()` | Add inline formulas |
