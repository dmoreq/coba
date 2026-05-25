# COBA Web Application — Comprehensive E2E Testing Plan

> **Date:** 2025-05-25
> **Coverage Target:** Pinpoint and eliminate every regression surface in UI rendering, state management, theme system, interaction loop, charting, navigation, and session lifecycle.

---

## 1. Current State Assessment

### 1.1 Test Inventory

| Directory | Files | Tests | Layer |
|-----------|-------|-------|-------|
| `tests/web/` | 7 files | ~53 tests | Theme tokens, event bus, interaction state, edge cases, route E2E |
| `tests/flet_redesign/` | 34 files | ~198 tests | Policy logic, math precision, simulator, worlds, curriculum, trace, session E2E |
| `tests/` (root core) | 27 files | ~350 tests | Backend bandit engine (not web) |
| **Total web-relevant** | **41 files** | **~251 tests** | — |

### 1.2 Coverage by Source Module

| Source Module | Status | Coverage |
|---|---|---|
| `statemgmt/event_bus.py` | ✅ Good | subscribe, emit, unsubscribe, error isolation, clear, async, all event names |
| `statemgmt/interaction_state.py` | ✅ Good | Phase ordering |
| `theme/constants.py` | ✅ Good | Spacing monotonic, font ordering, animation durations positive |
| `theme/tokens.py` | ✅ Good | Field parity, frozen, WCAG contrast ratio |
| `theme/theme_manager.py` | ❌ 0% | apply_theme, get_tokens, toggle — no mock-page tests |
| `components/agent.py` | ❌ 0% | knowledge_table, pull_counter, policy_state_card |
| `components/charts.py` | ❌ 0% | regret_chart, arm_histogram, reward_timeline |
| `components/environment.py` | ❌ 0% | world_card, context_display |
| `components/interaction.py` | ❌ 0% | arm_cards, reward_feedback, loop_visualizer, step_controls |
| `components/shared.py` | ❌ 0% | section_header, metric_badge, empty_state |
| `components/theme_toggle.py` | ❌ 0% | Build path, on_click callback |
| `layouts/split_workspace.py` | ❌ 0% | _build_zone, build (entire 3-zone dashboard) |
| `app.py` (AppShell) | ❌ 0% | All 10 methods untested |
| `main.py` (_SimSession, main, run) | ⚠️ ~40% | E2E tests cover lifecycle indirectly; `main()` untested |
| `ui/models.py` (dataclasses) | ⚠️ Indirect | Covered via `build_route_ui_model()` |

### 1.3 Root Causes of Testing Gaps

1. **Flet-dependent code is not tested.** All 6 `components/*.py` files, `layouts/split_workspace.py`, `app.py`, and `theme_manager.py` import `flet as ft` — and the existing test strategy deliberately avoids Flet imports by using `try: import flet` guards. This is fixable by mocking the `ft` module entry points.

2. **AppShell has zero tests.** The root application shell (10 methods, 260 lines) is the most critical untested component. Every route, navigation click, theme toggle, and view render passes through it.

3. **ThemeManager methods are untested.** `apply_theme`, `get_tokens`, and `toggle` work with `page.data`, `page.theme`, and `page.theme_mode` — all mockable attributes on a plain Python object.

4. **Component widget builders are untested.** They all follow the same pattern: `def build_*(page, ...) → ft.Control`. A mock `page` object with `.data = {}` suffices for testing.

5. **Session lifecycle methods** (`_SimSession`) are tested indirectly via E2E but several methods (`do_play`, `do_pause`, sync_prefs return value, autoplay cancellation) have untested branches.

---

## 2. Testing Architecture

### 2.1 Principle: Mock `page`, avoid real Flet

Flet has no headless browser testing mode. All UI component tests must pass `mock_page` — a plain Python object with the attributes each component reads:

```python
class MockPage:
    theme_mode = None     # ft.ThemeMode or string
    theme = None          # ft.Theme
    dark_theme = None     # ft.Theme
    data = {}             # dict for token storage
    views = []            # list for AppShell
    route = "/"           # current route string
    session = None        # No longer used directly

    def go(self, target): self.route = target
    def update(self): pass
    def run_task(self, coro): pass
```

### 2.2 File Organization

```
tests/web/
├── __init__.py
├── fixtures.py                    # MockPage, MockSession, mock_prefs, sample_world_config
├── test_e2e_routes.py             # (existing) Route view models & orchestrator
├── test_web_edge_cases.py         # (existing) Event bus, preferences edge cases
│
├── test_app_shell.py              # 🆕 AppShell lifecycle, nav, routing, view rendering
├── test_theme_manager.py          # 🆕 ThemeManager apply/toggle/get_tokens
├── test_session_lifecycle.py      # 🆕 _SimSession do_play/pause, sync_prefs, autoplay
│
├── test_components_agent.py       # 🆕 knowledge_table, pull_counter, policy_state_card
├── test_components_charts.py      # 🆕 regret_chart, arm_histogram, reward_timeline
├── test_components_environment.py # 🆕 world_card, context_display
├── test_components_interaction.py # 🆕 arm_cards, reward_feedback, loop_visualizer, controls
├── test_components_shared.py      # 🆕 section_header, metric_badge, empty_state
├── test_components_theme_toggle.py# 🆕 theme toggle button
├── test_layout_split_workspace.py # 🆕 SplitWorkspaceLayout 3-zone rendering
│
├── test_statemgmt/                # (existing) Event bus & interaction phase
└── test_theme/                    # (existing) Token validation
```

---

## 3. Test Plans by Module

### 3.1 `theme/theme_manager.py` — `tests/web/test_theme_manager.py`

**Risk:** Theme application/toggling breaks silently on Flet version upgrades (we already hit the `page.session.set` → `page.data` migration).

| Test | What It Verifies |
|------|------------------|
| `apply_theme_light_stores_tokens_on_page_data` | After `apply_theme(page, "light")`, `page.data["__coba_color_tokens"]` is LIGHT_TOKENS |
| `apply_theme_dark_stores_tokens_on_page_data` | After `apply_theme(page, "dark")`, stored tokens are DARK_TOKENS |
| `get_tokens_returns_stored_tokens` | `get_tokens(page)` returns whatever `apply_theme` stored |
| `get_tokens_falls_back_to_light_when_data_is_none` | When `page.data` is not set, returns LIGHT_TOKENS |
| `get_tokens_falls_back_when_data_not_dict` | When `page.data = "garbage"`, returns LIGHT_TOKENS (not crash) |
| `get_tokens_falls_back_when_key_missing` | When `page.data = {"other": "stuff"}`, returns LIGHT_TOKENS |
| `toggle_light_to_dark` | `toggle(page)` with LIGHT tokens → DARK_TOKENS returned |
| `toggle_dark_to_light` | `toggle(page)` with DARK tokens → LIGHT_TOKENS returned |
| `toggle_updates_page_theme` | After toggle, `page.theme_mode` switches accordingly |
| `toggle_updates_page_data` | After toggle, `page.data["__coba_color_tokens"]` is the new token set |

**Edge cases:**
- Calling `toggle` before `apply_theme` (should start from LIGHT default)
- Rapid consecutive toggles (5 toggles in a row — final state correct)
- `page.data` is `None` during toggle

---

### 3.2 `components/shared.py` — `tests/web/test_components_shared.py`

**Risk:** Shared components are used in every zone header. Layout breaks if spacing/fonts/colors misrender.

| Test | What It Verifies |
|------|------------------|
| `section_header_renders_title` | `build_section_header(page, "Title", "#000")` returns an `ft.Column` with the title text |
| `section_header_renders_accent_stripe` | The header includes an `ft.Container` with the accent color background |
| `section_header_uses_theme_spacing` | Controls use `SpacingScale.XS` spacing |
| `metric_badge_shows_value_and_label` | `build_metric_badge(page, "Steps", "42")` renders both strings |
| `metric_badge_label_uses_muted_color` | The label text uses `tokens.text_muted` from ThemeManager |
| `empty_state_renders_message` | `build_empty_state(page, "No data")` shows the message |
| `empty_state_has_info_icon` | The empty state includes an `ft.Icon` |
| `empty_state_default_message` | Calling without `message` uses "Nothing to show yet." |

---

### 3.3 `components/environment.py` — `tests/web/test_components_environment.py`

| Test | What It Verifies |
|------|------------------|
| `world_card_renders_title_and_description` | `build_world_card(page, "RidePilot", "A ride-hailing world")` renders both texts |
| `world_card_title_uses_environment_accent` | Title text color matches `tokens.environment_accent` |
| `context_display_renders_all_features` | A context dict of 3 features produces 3 `ft.Row` controls |
| `context_display_skips_step_key` | The `"step"` key is excluded from display |
| `context_display_uses_feature_order_when_provided` | Features display in provided order, not dict insertion order |
| `context_display_shows_placeholder_for_empty_context` | Empty context dict renders "No context features" italic text |
| `context_display_handles_feature_order_without_matching_keys` | Graceful when feature_order includes keys not in context |

---

### 3.4 `components/agent.py` — `tests/web/test_components_agent.py`

| Test | What It Verifies |
|------|------------------|
| `knowledge_table_renders_all_arms` | 3 arms → 3 rows in the column |
| `knowledge_table_shows_estimated_mean_with_tilde` | Mean rewards formatted as `~0.720` (tilde prefix) |
| `knowledge_table_shows_dash_for_zero_pulls` | Arm with 0 pulls shows "—" instead of mean |
| `knowledge_table_shows_pull_count_in_parens` | Pull count displayed as `(8)` after mean |
| `knowledge_table_shows_empty_message_for_no_arms` | Empty arm list renders "No arm data yet" |
| `pull_counter_shows_bar_proportional_to_count` | Bar width scales with count/max_count ratio |
| `pull_counter_shows_empty_message_for_zero_pulls` | All zero pulls → "No pulls yet" |
| `pull_counter_shows_empty_message_for_no_arms` | Empty list → "No pulls yet" |
| `policy_state_card_shows_policy_id` | Renders policy_id with agent_accent color |
| `policy_state_card_renders_passed_parameters` | `{"epsilon": 0.1}` → shows "epsilon: 0.1" |
| `policy_state_card_defaults_to_empty_dict` | `policy_data=None` → renders only policy_id row |

---

### 3.5 `components/interaction.py` — `tests/web/test_components_interaction.py`

| Test | What It Verifies |
|------|------------------|
| `arm_cards_renders_all_arms` | 3 arms → 3 containers returned |
| `arm_cards_highlights_selected_arm_with_amber_border` | Selected arm has `tokens.agent_accent` border |
| `arm_cards_selected_arm_uses_glow_background` | Selected arm bg = `tokens.selected_glow` |
| `arm_cards_shows_predicted_scores_when_provided` | `predicted_scores=[0.7, 0.3, 0.5]` → displays in labels |
| `arm_cards_graceful_on_mismatched_scores_length` | Scores shorter than arms — no crash |
| `arm_cards_shows_placeholder_for_empty_arms` | Empty list → "No arms available" |
| `reward_feedback_renders_placeholder_for_none_reward` | `reward=None` → empty 40px container |
| `reward_feedback_shows_success_for_positive_reward` | `reward=1.0` → green check, "Success! ✓" |
| `reward_feedback_shows_failure_for_zero_reward` | `reward=0.0` → red cancel, "No reward ✗" |
| `reward_feedback_has_animation` | Container has `animate` property |
| `loop_visualizer_shows_all_four_phases` | All 4 numbered circles + arrows rendered |
| `loop_visualizer_highlights_active_phase` | Active phase dot has accent color, others muted |
| `loop_visualizer_shows_idle_state_with_no_highlight` | `IDLE` phase — all dots muted |
| `loop_visualizer_renders_arrows_between_phases` | 3 arrow icons between 4 phases |
| `step_controls_renders_step_play_reset_buttons` | All three buttons present |
| `step_controls_play_shows_pause_label_when_running` | `is_running=True` → "⏸ Pause" |
| `step_controls_step_button_shows_current_count` | `current_step=42` → "Step (42)" |

---

### 3.6 `components/charts.py` — `tests/web/test_components_charts.py`

| Test | What It Verifies |
|------|------------------|
| `regret_chart_builds_with_empty_points` | `points=None` → chart renders without crash |
| `regret_chart_trims_to_max_points` | 120 points → only last 100 rendered |
| `regret_chart_uses_theme_line_color` | Series color = `tokens.chart_line_primary` |
| `regret_chart_has_transparent_background` | `bgcolor = tokens.chart_bg` (transparent) |
| `regret_chart_is_animated` | `animate=True`, `animation_duration=300` |
| `arm_histogram_builds_with_empty_labels` | `labels=None` → chart renders without crash |
| `arm_histogram_bars_use_agent_accent_color` | Bar color = `tokens.agent_accent` |
| `arm_histogram_tooltips_show_arm_name_and_count` | `tooltip="Priority: 12 pulls"` |
| `arm_histogram_bottom_axis_shows_label_per_bar` | axis labels match provided labels |
| `reward_timeline_shows_last_20_points` | 30 rewards → only last 20 data points |
| `reward_timeline_uses_green_line` | Line color = `tokens.success_feedback` |
| `reward_timeline_y_range_is_0_to_1` | `min_y=0`, `max_y=1` |
| `reward_timeline_renders_empty_without_crash` | `rewards=None` → renders gracefully |

---

### 3.7 `components/theme_toggle.py` — `tests/web/test_components_theme_toggle.py`

| Test | What It Verifies |
|------|------------------|
| `theme_toggle_renders_icon_button` | Returns `ft.IconButton` |
| `theme_toggle_shows_dark_mode_icon_when_dark` | `page.theme_mode == DARK` → `ft.Icons.DARK_MODE` |
| `theme_toggle_shows_light_mode_icon_when_light` | `page.theme_mode == LIGHT` → `ft.Icons.LIGHT_MODE` |
| `theme_toggle_click_calls_theme_manager_toggle` | `on_click` handler triggers `ThemeManager.toggle(page)` |
| `theme_toggle_click_calls_page_update` | `on_click` handler calls `page.update()` |
| `theme_toggle_has_correct_tooltip` | Tooltip text = "Toggle dark mode" |

---

### 3.8 `layouts/split_workspace.py` — `tests/web/test_layout_split_workspace.py`

| Test | What It Verifies |
|------|------------------|
| `layout_renders_three_zones` | `build()` returns `ft.Column` with top row of 3 zones |
| `layout_top_row_has_expand_true` | The top `ft.Row` has `expand=True` |
| `layout_zone_ratios_are_1_2_1` | Left expand=1, center expand=2, right expand=1 |
| `layout_charts_section_is_expansion_tile` | Bottom section is `ft.ExpansionTile` |
| `layout_charts_initially_expanded_when_controls_provided` | Non-empty chart_controls → `initially_expanded=True` |
| `layout_environment_zone_has_teal_background` | Left zone bg = `tokens.environment_zone_bg` |
| `layout_agent_zone_has_amber_background` | Right zone bg = `tokens.agent_zone_bg` |
| `layout_zone_borders_use_surface_border_color` | All zones have 1px `tokens.surface_border` border |
| `layout_zone_headers_show_correct_titles` | "Environment", "Interaction", "Agent" |
| `layout_passes_empty_controls_gracefully` | All four control lists are `None`/empty |

---

### 3.9 `app.py` (AppShell) — `tests/web/test_app_shell.py`

**This is the highest-value test file.** AppShell is the root application shell with no direct tests.

| Test | What It Verifies |
|------|------------------|
| `init_wires_page_on_route_change` | Constructor sets `page.on_route_change = self._on_route_change` |
| `init_wires_page_on_disconnect` | Constructor sets `page.on_disconnect` (from main.py, not AppShell) |
| `init_stores_page_session_pref_store` | Constructor stores three references |
| `apply_theme_calls_theme_manager` | `apply_theme("dark")` → `ThemeManager.apply_theme(page, "dark")` |
| `build_nav_rail_has_five_destinations` | Returns `NavigationRail` with 5 items |
| `build_nav_rail_selects_correct_index_for_current_route` | `/arena` → selected_index=2 |
| `build_nav_rail_none_selected_for_unknown_route` | `/unknown` → selected_index=None (no crash) |
| `navigate_to_calls_page_go_with_correct_route` | `_navigate_to(3)` → `page.go("/sandbox")` |
| `navigate_to_handles_out_of_range_index` | `_navigate_to(99)` → falls back to `"/"` |
| `navigate_to_cancels_autoplay` | Sets `_autoplay_task = None` |
| `build_top_bar_shows_lesson_stage_when_lesson_active` | session has lesson_progress → "Stage 2/5" in header |
| `build_top_bar_includes_theme_toggle` | Theme toggle icon button in actions list |
| `render_view_builds_full_page_for_active_route` | Non-home route → `ft.View` with `ft.AppBar`, `ft.NavigationRail` |
| `render_view_shows_heading_and_description` | View contains heading + description text |
| `render_view_home_route_skips_layout` | Home route → heading + description only, no 3-zone layout |
| `build_layout_provides_scene_panel_to_environment_zone` | Environment zone gets world title + description |
| `build_layout_provides_treatment_cards_to_interaction_zone` | Interaction zone gets arm card list |
| `build_layout_provides_lesson_panel_to_agent_zone` | Agent zone gets lesson title + stage when available |
| `refresh_view_calls_build_route_ui_model` | Triggers view model construction from session state |
| `refresh_view_handles_exceptions_gracefully` | Exception during build → prints traceback, does not crash |
| `on_route_change_calls_refresh_view` | Route change triggers full refresh |
| `cancel_autoplay_cancels_running_task` | Running `_autoplay_task` → cancelled, set to None |
| `cancel_autoplay_noop_when_no_task` | `_autoplay_task` is None → no error |

---

### 3.10 `main.py` (_SimSession) — `tests/web/test_session_lifecycle.py`

| Test | What It Verifies |
|------|------------------|
| `session_init_creates_simulator_and_controller` | After `__init__`, `simulator` and `controller` exist |
| `session_init_loads_lesson_when_policy_mapped` | UCB1 policy → `lesson_config` is not None |
| `session_init_graceful_when_policy_not_in_lesson_registry` | gp_ucb (no lesson) → `lesson_config = None`, no crash |
| `do_step_advances_simulator_by_one` | After `do_step`, `simulator.state.current_step` incremented |
| `do_step_from_idle_plays_then_pauses` | `controller.state.mode` transitions: idle → running → paused |
| `do_step_from_running_just_steps` | `controller.state.mode == "running"` → stays running after step |
| `do_play_sets_controller_to_running` | `do_play()` → `controller.state.mode == "running"` |
| `do_play_clears_cancel_flag` | `do_play()` → `_cancel_autoplay = False` |
| `do_pause_sets_controller_to_paused` | `do_pause()` → `controller.state.mode == "paused"` |
| `do_pause_sets_cancel_flag` | `do_pause()` → `_cancel_autoplay = True` |
| `do_reset_resets_simulator_to_step_zero` | State cleared, step=0, trace empty |
| `do_reset_resets_controller_to_idle` | `controller.state.mode == "idle"` |
| `do_reset_resets_lesson_progress_to_stage_1` | `lesson_progress.current_stage == 1` |
| `do_reset_noop_for_none_lesson` | Policy without lesson → no lesson_progress to reset |
| `sync_prefs_returns_true_when_world_changed` | Different world_id → returns `True` and rebuilds simulator |
| `sync_prefs_returns_false_when_world_same` | Same world_id → returns `False`, no simulator rebuild |
| `sync_prefs_returns_true_when_policy_changed` | Different policy_id → returns `True` |
| `session_simulator_is_rebuilt_on_world_change` | Post-rebuild, simulator.world.config.world_id matches new prefs |
| `session_simulator_is_rebuilt_on_policy_change` | Post-rebuild, policy type changes |

---

### 3.11 Consolidate Duplicated Tests

| Action | Rationale |
|--------|-----------|
| Move `test_web_edge_cases.py` route tests (view_model_with_missing_world_id, preferences defaults, CATS param controls) → appropriate component files or `test_e2e_routes.py` | Edge cases scattered across 2 files |
| Merge `test_statemgmt/test_interaction_state.py` (1 test) → `test_statemgmt/test_event_bus.py` | Single-function file, not worth separate module |
| Remove `tests/app/` (7 empty dirs with only `__pycache__`) | Dead test infrastructure from previous architecture |
| Ensure `test_e2e.py` (flet_redesign) and `test_e2e_routes.py` (web) don't test the same things | Both test route view models and orchestrator; consolidate overlap |

---

## 4. E2E Session Lifecycle Testing

These tests verify the full user journey end-to-end, from page load through simulation, reset, and navigation.

### 4.1 Session Initialization Flow

| Test | Steps |
|------|-------|
| `cold_start_with_default_preferences` | Load preferences → create session → verify world="rural_clinic", policy="random", step=0, mode="idle" |
| `cold_start_restores_saved_preferences` | Save prefs (world="ridepilot", policy="ucb1") → reload → session uses those prefs |
| `cold_start_with_corrupted_prefs_file` | Corrupted JSON → falls back to defaults without crash |

### 4.2 Simulation Interaction Flow

| Test | Steps |
|------|-------|
| `step_by_step_full_cycle` | Step 5 times → verify step count=5, trace has 5 entries, cumulative values accumulate |
| `play_autoplay_runs_multiple_steps` | Play → wait 5 cycles → Pause → verify step count > 5 |
| `play_reset_during_autoplay_clears_state` | Play → step > 0 → Reset while running → step=0, trace empty, controller idle |
| `run_fifty_steps_from_idle` | Run 50 steps → verify monotonic step indices, 50 trace entries |
| `world_switch_during_autoplay_cancels_and_rebuilds` | Play → Change world → autoplay cancelled, simulator rebuilt with new world |
| `policy_switch_during_autoplay_cancels_and_rebuilds` | Play → Change policy → autoplay cancelled, simulator rebuilt with new policy |

### 4.3 Navigation Flow

| Test | Steps |
|------|-------|
| `navigate_all_five_routes_does_not_crash` | Visit / → /lesson → /arena → /sandbox → /comparison → verify each route renders |
| `navigate_preserves_simulation_state` | Run 10 steps on /arena → navigate to /lesson → back to /arena → step still 10 |
| `navigate_unknown_route_falls_back_to_home` | Visit /garbage → route resolves to "/" |
| `lesson_route_shows_stage_indicator` | Visit /lesson with UCB1 → top bar shows "Stage 1/5" |
| `lesson_advances_stage_after_objective_met` | Run enough steps to meet objective → stage advances |

### 4.4 Theme Toggle Flow

| Test | Steps |
|------|-------|
| `theme_toggle_switches_light_to_dark` | Toggle → page.theme_mode=DARK, tokens switch to DARK_TOKENS |
| `theme_toggle_switches_dark_to_light` | Dark → Toggle → page.theme_mode=LIGHT, tokens switch to LIGHT_TOKENS |
| `theme_persists_across_navigation` | Toggle to dark → navigate to /arena → tokens still DARK_TOKENS |
| `theme_toggle_updates_chart_colors` | Toggle → chart line colors, grid colors, bar colors all update to dark equivalents |

---

## 5. Edge Case Coverage Expansion

### 5.1 State Boundaries

| Test | Scenario |
|------|----------|
| `step_beyond_simulator_horizon` | Run 10000 steps on horizon=100 → no crash, step=10000 |
| `step_at_exact_horizon_boundary` | Run exactly horizon steps → step=horizon, all trace entries present |
| `reset_immediately_after_step_then_step_again` | Step → reset → step → step → verify state=2 (no corruption) |
| `double_reset_is_idempotent` | Step → reset → reset → step=0, trace=0 |
| `play_then_play_again_is_idempotent` | Play → Play → controller still "running" |
| `pause_then_pause_again_is_idempotent` | Pause → Pause → controller still "paused" |
| `empty_arms_on_contextual_policy` | LinUCB with empty arms list → raises ValueError |
| `zero_feature_contextual_policy` | LinUCB with feature_order=() → handles gracefully |

### 5.2 Data Integrity

| Test | Scenario |
|------|----------|
| `trace_indices_are_monotonic_after_rapid_steps` | 50 rapid steps → step_index 1,2,...,50 |
| `trace_indices_reset_after_session_reset` | 10 steps → reset → 5 steps → indices 1,2,3,4,5 |
| `cumulative_reward_never_decreases` | Run 100 steps → verify cumulative_reward monotonic non-decreasing |
| `regret_with_perfect_policy_is_zero` | Policy always picks optimal → regret sum = 0 |
| `trace_json_produces_valid_utf8` | Serialized trace → `json.loads` → roundtrip matches |
| `trace_csv_has_correct_column_count` | CSV export → 7 columns in header |
| `trace_csv_empty_produces_empty_string` | Trace with 0 entries → `""` |

### 5.3 Error Handling

| Test | Scenario |
|------|----------|
| `build_route_ui_model_with_missing_world` | world_id="nonexistent" → raises KeyError |
| `build_route_ui_model_with_missing_policy` | policy_id="nonexistent" → raises KeyError |
| `unknown_policy_capability_lookup` | get_policy_capability("unknown") → raises KeyError |
| `world_arm_count_below_two` | World with 1 arm → schema validation raises ValueError |
| `world_feature_has_duplicate_names` | Two features with same name → ValueError |
| `arm_weight_references_nonexistent_feature` | Weight on feature not in config → ValueError |

---

## 6. Implementation Order (Priority by Risk)

| Priority | Test File | Risk Level | Why |
|----------|-----------|-----------|-----|
| **P0** | `test_app_shell.py` | 🔴 Critical | AppShell is untested root component; all routing/pages go through it |
| **P1** | `test_session_lifecycle.py` | 🔴 Critical | _SimSession lifecycle methods have untested branches (do_play, do_pause, sync_prefs, lesson reset) |
| **P1** | `test_theme_manager.py` | 🟠 High | Theme broke on Flet 0.85.1; needs regression tests |
| **P2** | `test_components_interaction.py` | 🟠 High | Arm cards, reward feedback, loop visualizer — core interaction UX |
| **P2** | `test_components_charts.py` | 🟡 Medium | Charts not rendering shows as empty text; silent failure |
| **P2** | `test_components_agent.py` | 🟡 Medium | Knowledge table, pull counter — core agent visualization |
| **P3** | `test_components_environment.py` | 🟡 Medium | World card, context display |
| **P3** | `test_layout_split_workspace.py` | 🟡 Medium | 3-zone dashboard layout |
| **P3** | `test_components_shared.py` | 🟢 Low | Utility components used in every zone header |
| **P3** | `test_components_theme_toggle.py` | 🟢 Low | Single button component |
| **P4** | Consolidate duplicate tests | 🟢 Low | Cleanup, not new coverage |
| **P4** | Delete `tests/app/` empty dirs | 🟢 Low | Dead test infra cleanup |

---

## 7. Test Execution & Verification

### After Each Priority Level

```bash
# Run the new test file
uv run pytest tests/web/test_app_shell.py -v -p no:asyncio

# Full suite (expect no regressions)
uv run pytest tests/ -v -p no:asyncio --ignore=tests/test_shared_sim.py

# With coverage
uv run pytest tests/flet_redesign/ tests/web/ --cov=src/web \
    --cov-report=term-missing -p no:asyncio
```

### Commit After Each File

```
test(web): add AppShell routing, navigation, and view rendering tests
test(web): add _SimSession lifecycle tests (play/pause/reset/sync_prefs)
test(web): add ThemeManager apply/toggle/get_tokens tests
test(web): add interaction component tests (arm cards, feedback, loop)
...
```

---

## 8. Expected Final State

| Metric | Current | Target |
|--------|---------|--------|
| Web test files | 7 | 17 |
| Web test count | ~53 | ~180+ |
| Component coverage (widget builders) | 0% | >85% |
| AppShell coverage | 0% | >90% |
| Session lifecycle coverage | ~40% | >90% |
| Theme manager coverage | 0% | 100% |
| Dead test dirs | 7 | 0 |
| Duplicate test patterns | 3 | 0 |
