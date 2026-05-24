"""Flet app entrypoint — interactive simulation with live state and lesson progression."""

from __future__ import annotations

import asyncio
from typing import Any

from web.curriculum import (
    LessonProgressState,
    evaluate_lesson_objective,
    get_lesson_by_policy,
    locked_control_keys_for_stage,
)
from web.policy_factory import build_policy
from web.simulator import DiscreteSimulator
from web.state import RunConfig
from web.ui.charts import build_chart_data
from web.ui.components.trace_table import build_trace_table
from web.ui.preferences import PreferencesStore, UserPreferences
from web.ui.run_controls import RunController
from web.ui.view_models import RouteUIModel, build_route_ui_model
from web.worlds import create_world, get_world_config, list_world_configs

try:
    import flet as ft
except ModuleNotFoundError:  # pragma: no cover
    ft = None  # type: ignore[assignment]


class _SimSession:
    """Mutable session state that lives across Flet route changes.

    Stores the live simulator, run controller, lesson progress, and
    the auto-play cancellation flag so the Flet shell can step, pause,
    and reset without re-creating the engine each render cycle.
    """

    def __init__(self, prefs: UserPreferences) -> None:
        self.prefs = prefs
        self._reset_simulator()

    def _reset_simulator(self) -> None:
        world = create_world(self.prefs.world_id)
        config = get_world_config(self.prefs.world_id)
        feature_order = tuple(f.name for f in config.features)
        policy = build_policy(
            self.prefs.policy_id,
            feature_order=feature_order,
            seed=0,
        )
        self.simulator = DiscreteSimulator(
            policy=policy,
            world=world,
            config=RunConfig(seed=0, horizon=10000),
        )
        self.simulator.reset()
        self.controller = RunController()
        self._cancel_autoplay = False

        lesson = None
        try:
            lesson = get_lesson_by_policy(self.prefs.policy_id)
        except KeyError:
            pass
        self.lesson_config = lesson
        self.lesson_progress = (
            LessonProgressState(lesson_id=lesson.lesson_id, current_stage=1, completed=False)
            if lesson
            else None
        )

    def sync_prefs(self, prefs: UserPreferences) -> bool:
        """Apply preference changes. Returns True if simulator was rebuilt."""
        changed = prefs.world_id != self.prefs.world_id or prefs.policy_id != self.prefs.policy_id
        self.prefs = prefs
        if changed:
            self._reset_simulator()
        return changed

    def do_step(self) -> None:
        if self.controller.state.mode == "running":
            self.controller.step()
        elif self.controller.state.mode in ("idle", "paused"):
            self.controller.play()
            self.controller.step()
            self.controller.pause()

    def do_play(self) -> None:
        self.controller.play()
        self._cancel_autoplay = False

    def do_pause(self) -> None:
        self.controller.pause()
        self._cancel_autoplay = True

    def do_reset(self) -> None:
        self._cancel_autoplay = True
        self.simulator.reset()
        self.controller.reset()
        if self.lesson_config:
            self.lesson_progress = LessonProgressState(
                lesson_id=self.lesson_config.lesson_id,
                current_stage=1,
                completed=False,
            )


def _build_view(
    session: _SimSession,
    route: str | None,
) -> RouteUIModel:
    sim = session.simulator
    records = tuple(sim.trace_buffer.to_records())
    context = sim.world.sample_context(sim.state.current_step + 1)
    disabled_keys: tuple[str, ...] = ()
    if session.lesson_config and session.lesson_progress:
        stage = session.lesson_progress.current_stage
        disabled_keys = locked_control_keys_for_stage(session.lesson_config, stage=stage)

    return build_route_ui_model(
        route,
        prefs=session.prefs,
        trace_records=records,
        sim_context=context,
        lesson_progress=session.lesson_progress,
        lesson_config=session.lesson_config,
        sim_step_index=sim.state.current_step,
        sim_cumulative_reward=sim.state.cumulative_reward,
        sim_cumulative_regret=sim.state.cumulative_regret,
        disabled_control_keys=disabled_keys,
    )


def _advance_lesson_if_ready(session: _SimSession) -> bool:
    """Check whether lesson objectives are met and advance stage if so."""
    if not session.lesson_config or not session.lesson_progress:
        return False
    if session.lesson_progress.completed:
        return False

    obj = session.lesson_config.objective
    sim = session.simulator
    if evaluate_lesson_objective(
        objective=obj,
        steps_executed=sim.state.current_step,
        cumulative_reward=sim.state.cumulative_reward,
        cumulative_regret=sim.state.cumulative_regret,
    ):
        progress = session.lesson_progress.advance()
        if progress.current_stage > 5:
            progress = progress.mark_completed()
        session.lesson_progress = progress
        return True
    return False


def _render_nav_rail(current_route: str) -> Any:
    if ft is None:
        return None
    return ft.NavigationRail(
        selected_index=None,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=100,
        min_extended_width=160,
        destinations=[
            ft.NavigationRailDestination(
                icon=ft.Icons.HOME_OUTLINED,
                selected_icon=ft.Icons.HOME,
                label="Home",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.SCHOOL_OUTLINED,
                selected_icon=ft.Icons.SCHOOL,
                label="Lesson",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.ANALYTICS_OUTLINED,
                selected_icon=ft.Icons.ANALYTICS,
                label="Arena",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.SCIENCE_OUTLINED,
                selected_icon=ft.Icons.SCIENCE,
                label="Sandbox",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.COMPARE_ARROWS_OUTLINED,
                selected_icon=ft.Icons.COMPARE_ARROWS,
                label="Compare",
            ),
        ],
        on_change=lambda e: _navigate_to(e.control.selected_index),
    )


def _render_world_selector(current_world: str) -> Any:
    if ft is None:
        return None
    worlds = list_world_configs()
    return ft.Dropdown(
        value=current_world,
        options=[ft.dropdown.Option(w.world_id, w.title) for w in worlds],
        width=200,
        on_select=lambda e: _on_world_change(e.control.value),
    )


def _render_policy_selector(current_policy: str) -> Any:
    if ft is None:
        return None
    policy_labels: dict[str, str] = {
        "random": "Random",
        "epsilon_greedy": "Epsilon-Greedy",
        "ucb1": "UCB1",
        "thompson": "Thompson",
        "softmax": "Softmax",
        "linucb": "LinUCB",
        "linucb_sw": "LinUCB-SW",
        "lints": "LinTS",
        "logistic_ucb": "Logistic UCB",
        "gp_ucb": "GP-UCB",
        "bootstrapped_ensemble": "Bootstrap Ensemble",
        "linucb_hybrid": "LinUCB Hybrid",
        "tree_ucb": "Tree UCB",
        "tree_ts": "Tree TS",
        "cats": "CATS",
    }
    options = [
        ft.dropdown.Option(key=k, text=f"{label} ({k})") for k, label in policy_labels.items()
    ]
    return ft.Dropdown(
        value=current_policy,
        options=options,
        width=240,
        on_select=lambda e: _on_policy_change(e.control.value),
    )


def _render_speed_selector(current_speed: str) -> Any:
    if ft is None:
        return None
    speeds = ["0.25x", "0.5x", "1x", "2x", "4x", "8x"]
    return ft.Dropdown(
        value=current_speed,
        options=[ft.dropdown.Option(s) for s in speeds],
        width=100,
        on_select=lambda e: _on_speed_change(e.control.value),
    )


def _render_shell_view(view: RouteUIModel, session: _SimSession) -> Any:
    if ft is None:
        raise RuntimeError("Flet is not installed.")

    is_active_route = view.layout is not None and view.scene_panel is not None
    header_controls: list[Any] = [
        _render_world_selector(session.prefs.world_id),
        _render_policy_selector(session.prefs.policy_id),
        _render_speed_selector(session.prefs.speed),
    ]
    if is_active_route and session.lesson_config:
        header_controls.insert(
            0,
            ft.Text(
                (
                    f"Stage {session.lesson_progress.current_stage}/5"
                    if session.lesson_progress
                    else ""
                ),
                italic=True,
            ),
        )

    content_panels: list[Any] = [ft.Text(value=view.heading, size=28, weight=ft.FontWeight.BOLD)]
    if not view.layout or view.scene_panel is None:
        content_panels.append(ft.Text(value=view.description, size=14))
    else:
        content_panels.append(ft.Text(value=view.description, size=14))
        content_panels.append(_build_three_pane_body(view, session))

    return ft.View(
        route=view.route,
        controls=[
            ft.AppBar(
                title=ft.Text(f"COBA · {view.title}"),
                actions=header_controls,
            ),
            ft.Row(
                controls=[
                    _render_nav_rail(view.route),
                    ft.VerticalDivider(width=1),
                    ft.Container(
                        content=ft.Column(
                            controls=content_panels,
                            spacing=10,
                            scroll=ft.ScrollMode.AUTO,
                        ),
                        padding=20,
                        expand=True,
                    ),
                ],
                expand=True,
            ),
        ],
    )


def _build_control_row(session: _SimSession) -> Any:
    if ft is None:
        return None
    mode = session.controller.state.mode
    step_label = f"  Step ({session.simulator.state.current_step})  "
    play_label = "  ▶ Play  " if mode != "running" else "  ⏸ Pause  "

    return ft.Row(
        controls=[
            ft.Button(text=step_label, on_click=lambda _: _on_step()),
            ft.Button(text=play_label, on_click=lambda _: _on_play()),
            ft.Button(text="  ↺ Reset  ", on_click=lambda _: _on_reset()),
        ],
        spacing=8,
    )


def _build_three_pane_body(view: RouteUIModel, session: _SimSession) -> Any:
    if ft is None:
        return None

    left_panel = ft.Container(
        expand=1,
        padding=12,
        border=ft.Border(
            left=ft.BorderSide(1, "#D5D7DA"),
            top=ft.BorderSide(1, "#D5D7DA"),
            right=ft.BorderSide(1, "#D5D7DA"),
            bottom=ft.BorderSide(1, "#D5D7DA"),
        ),
        border_radius=8,
        content=ft.Column(
            controls=[
                ft.Text(value=view.layout.left.title, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                ft.Text(value=view.scene_panel.world_title, weight=ft.FontWeight.W_600),
                ft.Text(value=view.scene_panel.world_description, size=12),
                ft.Divider(),
                ft.Text("Context", size=11, italic=True),
            ]
            + [
                ft.Text(value=f"{key}: {value}", size=11)
                for key, value in view.scene_panel.context_items.items()
                if key != "step"
            ],
            tight=True,
            spacing=4,
        ),
    )

    center_controls: list[Any] = [
        ft.Text(value=view.layout.center.title, weight=ft.FontWeight.BOLD),
        ft.Divider(),
        _build_control_row(session),
        ft.Divider(),
    ]
    for card in view.treatment_cards:
        suffix = "  ← selected" if card.selected else ""
        center_controls.append(
            ft.Text(value=f"{card.label}{suffix}", size=14, weight=ft.FontWeight.W_500)
        )
    center_panel = ft.Container(
        expand=2,
        padding=12,
        border=ft.Border(
            left=ft.BorderSide(1, "#D5D7DA"),
            top=ft.BorderSide(1, "#D5D7DA"),
            right=ft.BorderSide(1, "#D5D7DA"),
            bottom=ft.BorderSide(1, "#D5D7DA"),
        ),
        border_radius=8,
        content=ft.Column(controls=center_controls, spacing=4, scroll=ft.ScrollMode.AUTO),
    )

    right_controls: list[Any] = [
        ft.Text(value=view.layout.right.title, weight=ft.FontWeight.BOLD),
        ft.Divider(),
    ]
    sim = session.simulator
    right_controls.append(
        ft.Text(f"Steps: {sim.state.current_step}", size=12, weight=ft.FontWeight.BOLD)
    )
    right_controls.append(ft.Text(f"Cum. Reward: {sim.state.cumulative_reward:.3f}", size=12))
    right_controls.append(ft.Text(f"Cum. Regret: {sim.state.cumulative_regret:.3f}", size=12))

    if view.arena_metrics is not None:
        sim = session.simulator
        records = list(sim.trace_buffer.to_records())
        chart_data = build_chart_data(view.arena_metrics)
        trace_model = build_trace_table(records)
        right_controls.append(ft.Divider())
        right_controls.append(
            ft.Text(f"Steps: {sim.state.current_step}", size=11, weight=ft.FontWeight.BOLD)
        )
        right_controls.append(ft.Text(f"Cum. Reward: {sim.state.cumulative_reward:.3f}", size=11))
        right_controls.append(ft.Text(f"Cum. Regret: {sim.state.cumulative_regret:.3f}", size=11))
        right_controls.append(ft.Text(f"Trace entries: {trace_model.total_entries}", size=11))
        pull_str = ", ".join(
            f"{arm}: {cnt}" for arm, cnt in view.arena_metrics.arm_pull_counts.items()
        )
        right_controls.append(ft.Text(f"Pulls: {pull_str}", size=11))
        if chart_data.reward_points:
            right_controls.append(ft.Text("Reward (last 10):", size=10, italic=True))
            last_pts = chart_data.reward_points[-10:]
            reward_line = ", ".join(f"({s},{v:.2f})" for s, v in last_pts)
            right_controls.append(ft.Text(reward_line, size=9))

    if view.lesson_panel is not None:
        right_controls.append(ft.Divider())
        right_controls.append(
            ft.Text(f"Lesson: {view.lesson_panel.lesson_title}", size=12, weight=ft.FontWeight.BOLD)
        )
        right_controls.append(ft.Text(f"Stage: {view.lesson_panel.stage_index}/5", size=12))
        right_controls.append(
            ft.Container(
                content=ft.Markdown(view.lesson_panel.theory_markdown, selectable=True),
                padding=ft.Padding(top=4, bottom=4, left=0, right=0),
            )
        )
        right_controls.append(ft.Text(view.lesson_panel.objective_text, size=12, color="#D84315"))
        right_controls.append(ft.Text(view.lesson_panel.step_explanation, size=10, italic=True))

    if view.context_inspection is not None:
        right_controls.append(ft.Divider())
        right_controls.append(ft.Text("Feature Vector", size=11, weight=ft.FontWeight.BOLD))
        vec_str = ", ".join(
            f"{n}={v:.3f}"
            for n, v in zip(
                view.context_inspection.feature_order,
                view.context_inspection.feature_values,
            )
        )
        right_controls.append(ft.Text(vec_str, size=11))

    right_controls.append(ft.Divider())
    right_controls.append(
        ft.Text(f"Debug: {', '.join(view.capability_debug_views) or 'summary'}", size=11)
    )

    if view.param_controls:
        right_controls.append(ft.Text("Parameters", size=11, weight=ft.FontWeight.BOLD))
        for spec in view.param_controls:
            disabled = spec.key in view.locked_controls
            label_suffix = " 🔒" if disabled else ""
            if spec.control_type == "slider":
                right_controls.append(
                    ft.Text(
                        f"{spec.label}{label_suffix}: {spec.default_value}",
                        size=11,
                    )
                )
            else:
                right_controls.append(
                    ft.Text(f"{spec.label}{label_suffix}: {spec.default_value}", size=11)
                )

    right_panel = ft.Container(
        expand=1,
        padding=12,
        border=ft.Border(
            left=ft.BorderSide(1, "#D5D7DA"),
            top=ft.BorderSide(1, "#D5D7DA"),
            right=ft.BorderSide(1, "#D5D7DA"),
            bottom=ft.BorderSide(1, "#D5D7DA"),
        ),
        border_radius=8,
        content=ft.Column(
            controls=right_controls,
            spacing=4,
            scroll=ft.ScrollMode.AUTO,
        ),
    )

    return ft.Row(
        controls=[left_panel, center_panel, right_panel],
        spacing=10,
        expand=True,
    )


# ── Global state refs (set in main()) ──────────────────────────────
_page: Any = None
_session: _SimSession | None = None
_pref_store: PreferencesStore | None = None
_autoplay_task: asyncio.Task[None] | None = None


def _refresh_view() -> None:
    if _page is None or _session is None:
        return
    route = _page.route or "/"
    try:
        view = _build_view(_session, route)
        _page.views.clear()
        _page.views.append(_render_shell_view(view, _session))
        _page.update()
    except Exception:
        import traceback

        traceback.print_exc()


def _on_step() -> None:
    if _session is None:
        return
    _session.do_step()
    sim = _session.simulator
    sim.step()
    _advance_lesson_if_ready(_session)
    _session.controller.pause()
    _refresh_view()


async def _autoplay_loop() -> None:
    while _session is not None and not _session._cancel_autoplay:
        sim = _session.simulator
        sim.step()
        _session.controller.step()
        _advance_lesson_if_ready(_session)
        _refresh_view()
        speed = _session.prefs.speed
        try:
            multiplier = float(speed.replace("x", ""))
        except (ValueError, AttributeError):
            multiplier = 1.0
        delay = max(0.02, 1.0 / multiplier * 0.5)
        await asyncio.sleep(delay)


def _on_play() -> None:
    global _autoplay_task
    if _session is None:
        return
    if _session.controller.state.mode == "running":
        _session.do_pause()
        if _autoplay_task:
            _autoplay_task.cancel()
            _autoplay_task = None
        _refresh_view()
    else:
        _session.do_play()
        _refresh_view()
        if _page is not None:
            _autoplay_task = _page.run_task(_autoplay_loop)


def _on_reset() -> None:
    global _autoplay_task
    if _session is None:
        return
    if _autoplay_task:
        _autoplay_task.cancel()
        _autoplay_task = None
    _session.do_reset()
    _refresh_view()


def _navigate_to(index: int) -> None:
    if _page is None:
        return
    routes = ["/", "/lesson", "/arena", "/sandbox", "/comparison"]
    target = routes[index] if 0 <= index < len(routes) else "/"
    _page.go(target)


def _on_world_change(world_id: str) -> None:
    global _autoplay_task
    if _session is None or not world_id:
        return
    if _autoplay_task:
        _autoplay_task.cancel()
        _autoplay_task = None
    prefs = UserPreferences(
        world_id=world_id,
        policy_id=_session.prefs.policy_id,
        speed=_session.prefs.speed,
    )
    _session.sync_prefs(prefs)
    if _pref_store:
        _pref_store.save(prefs)
    _refresh_view()


def _on_policy_change(policy_id: str) -> None:
    global _autoplay_task
    if _session is None or not policy_id:
        return
    if _autoplay_task:
        _autoplay_task.cancel()
        _autoplay_task = None
    prefs = UserPreferences(
        world_id=_session.prefs.world_id,
        policy_id=policy_id,
        speed=_session.prefs.speed,
    )
    _session.sync_prefs(prefs)
    if _pref_store:
        _pref_store.save(prefs)
    _refresh_view()


def _on_speed_change(speed: str) -> None:
    if _session is None:
        return
    prefs = UserPreferences(
        world_id=_session.prefs.world_id,
        policy_id=_session.prefs.policy_id,
        speed=speed,
    )
    _session.prefs = prefs
    if _pref_store:
        _pref_store.save(prefs)


def main(page: Any) -> None:
    global _page, _session, _pref_store
    if ft is None:
        raise RuntimeError("Flet is not installed.")

    _page = page
    _pref_store = PreferencesStore()
    prefs = _pref_store.load()
    _session = _SimSession(prefs)

    page.title = "COBA — Contextual Bandit Lab"

    def on_route_change(event: Any) -> None:
        _refresh_view()

    def on_disconnect(event: Any) -> None:
        _ = event
        global _autoplay_task
        if _autoplay_task:
            _autoplay_task.cancel()
            _autoplay_task = None
        if _pref_store and _session:
            _pref_store.save(_session.prefs)

    page.on_route_change = on_route_change
    page.on_disconnect = on_disconnect

    _refresh_view()
    page.update()


def run() -> None:
    if ft is None:
        raise RuntimeError("Flet is not installed. Install it with `pip install flet`.")
    ft.run(main, view=ft.AppView.WEB_BROWSER, port=8550)


if __name__ == "__main__":
    run()
