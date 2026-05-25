# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- Theme system: ColorTokens dataclass, ThemeManager, dark/light mode toggle
- SplitWorkspaceLayout: 3-zone dashboard (environment/interaction/agent + bottom charts)
- AppShell: navigation rail, route dispatch, view rendering
- EventBus: pub/sub for cross-component communication
- InteractionPhase: 4-phase animation state enum
- Environment zone components: world_card, context_display
- Agent zone components: knowledge_table (~ prefix), pull_counter (bar chart), policy_state_card
- Interaction zone components: arm_cards (amber glow on selected), reward_feedback (green/red), loop_visualizer (4 numbered phases), step_controls
- Chart components: regret_chart, arm_histogram, reward_timeline (all built from ft.Container bars for Flet 0.85.1 compat)
- Edge case tests for event bus, view models, preferences, param controls

### Fixed
- Flet 0.85.1 API incompatibilities: page.session.set/get → page.data
- Flet 0.85.1: removed nonexistent ft.animation.Animation, ft.LineChart/ft.BarChart
- Flet 0.85.1 ExpansionTile: initially_expanded → expanded
- Interaction loop phase comparison logic (was always false)
- build_reward_feedback syntax error (duplicate return block)

### Removed
- Dead code: shell.py, state_store.py, preset_manager.py, checkpoint.py
- Dead code: drift/ (3 files — local CUSUM/ADWIN, zero callers)
- Dead code: debug/context_free.py, debug/continuous.py
- Dead code: comparison/snapshot_diff.py (diff_trace_records/diff_debug_snapshots, zero callers)
- Dead code: tests/flet_redesign/test_shell.py
- 8 thin dataclass files consolidated into ui/models.py
- arena/ + comparison/ merged into analysis/
- ui/pages/comparison_page.py, sandbox_page.py (logic consolidated into view_models.py)
- Stale docs: coba-redesign-plan.md, option-c-implementation-plan.md, e2e-testing-plan.md, web_testing_plan.md, flet_redesign_*.md, web_app_plan.md, web_implementation_plan.md, web_architecture.md
- docs/superpowers/ (outdated design specs)
- graphify-out/GRAPH_REPORT.md

### Changed
- Redesigned main.py: delegates to AppShell instead of monolithic rendering
- All hardcoded hex colors consolidated into theme/tokens.py
- All components read from ThemeManager.get_tokens(page) instead of inline values

---

## [1.0.0] — 2026-05-18

### Web Platform

#### Added
- 17 interactive lessons (exploration vs exploitation, UCB1, Thompson Sampling, LinUCB, LinTS, Softmax, Sliding-Window LinUCB, Logistic UCB, GP-UCB, Random Forest, Bootstrapped Ensemble, LinUCB Hybrid, Offline Evaluation, Drift Detection, Production Features)
- Real-time visualizations using Recharts (cumulative reward, regret, arm scores, distributions, cluster map)
- Algorithm theory cards with collapsible math explanations
- Keyboard navigation (Space, arrows, numbers, Ctrl+N/P)
- Dark mode support
- 103 frontend tests, 63 backend tests, 90% backend coverage, zero TypeScript errors
- CI/CD with GitHub Actions (auto-deploy to Vercel on main merge)
- Full responsive design (desktop, tablet, mobile)
