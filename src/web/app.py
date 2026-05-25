"""AppShell — root application shell for the COBA web app.

Handles page structure, navigation, theme, route dispatch, autoplay.
Session state is stored on page.data for multi-user web isolation.
"""

from __future__ import annotations

import asyncio
from typing import Any

from web.components.theme_toggle import build_theme_toggle
from web.curriculum import evaluate_lesson_objective
from web.layouts.split_workspace import SplitWorkspaceLayout
from web.session import get_session
from web.theme import FontScale, SpacingScale
from web.theme.theme_manager import ThemeManager
from web.ui.view_models import RouteUIModel, build_route_ui_model

try:
    import flet as ft
except ModuleNotFoundError:  # pragma: no cover
    ft = None  # type: ignore[assignment]

ROUTES = ["/", "/lesson", "/arena", "/sandbox", "/comparison"]
ROUTE_LABELS = ["Home", "Lesson", "Arena", "Sandbox", "Compare"]
ROUTE_ICONS = [
    ft.Icons.HOME_OUTLINED,
    ft.Icons.SCHOOL_OUTLINED,
    ft.Icons.ANALYTICS_OUTLINED,
    ft.Icons.SCIENCE_OUTLINED,
    ft.Icons.COMPARE_ARROWS_OUTLINED,
]
ROUTE_ICONS_SELECTED = [
    ft.Icons.HOME,
    ft.Icons.SCHOOL,
    ft.Icons.ANALYTICS,
    ft.Icons.SCIENCE,
    ft.Icons.COMPARE_ARROWS,
]


class AppShell:
    """Root application shell that manages navigation, theme, and rendering."""

    def __init__(self, page: Any) -> None:
        self.page = page
        self._autoplay_task: asyncio.Task[None] | None = None
        page.on_route_change = self._on_route_change

    # ── Session helpers ──────────────────────────────────────────────

    @property
    def session(self) -> Any:
        return get_session(self.page)

    # ── Navigation bar ───────────────────────────────────────────────

    def _build_nav_rail(self, current_route: str) -> Any:
        if ft is None:
            return None
        try:
            selected = ROUTES.index(current_route) if current_route in ROUTES else None
        except ValueError:
            selected = None

        tokens = ThemeManager.get_tokens(self.page)
        return ft.NavigationRail(
            selected_index=selected,
            label_type=ft.NavigationRailLabelType.ALL,
            min_width=100,
            min_extended_width=160,
            bgcolor=tokens.bg_secondary,
            destinations=[
                ft.NavigationRailDestination(
                    icon=ROUTE_ICONS[i],
                    selected_icon=ROUTE_ICONS_SELECTED[i],
                    label=ROUTE_LABELS[i],
                )
                for i in range(len(ROUTES))
            ],
            on_change=lambda e: self._navigate_to(e.control.selected_index),
        )

    def _navigate_to(self, index: int) -> None:
        target = ROUTES[index] if 0 <= index < len(ROUTES) else "/"
        self._cancel_autoplay()
        self.page.go(target)

    # ── Top bar / selectors ──────────────────────────────────────────

    def _build_top_bar(self, view: RouteUIModel) -> Any:
        if ft is None:
            return None

        actions: list[Any] = [build_theme_toggle(self.page)]

        sess = self.session
        if sess and sess.lesson_progress:
            actions.insert(
                0,
                ft.Text(
                    f"Stage {sess.lesson_progress.current_stage}/5",
                    italic=True,
                    size=FontScale.SMALL,
                ),
            )

        return ft.AppBar(
            title=ft.Text(f"COBA · {view.title}", size=FontScale.TITLE),
            actions=actions,
        )

    # ── View rendering ───────────────────────────────────────────────

    def _render_view(self, view: RouteUIModel) -> Any:
        """Build the full page view."""
        if ft is None:
            raise RuntimeError("Flet not installed")

        is_active = view.layout is not None and view.scene_panel is not None

        content_panels: list[Any] = [
            ft.Text(value=view.heading, size=FontScale.HEADING, weight=ft.FontWeight.BOLD),
        ]
        if not is_active:
            content_panels.append(ft.Text(value=view.description, size=FontScale.BODY))
        else:
            content_panels.append(ft.Text(value=view.description, size=FontScale.SMALL))
            content_panels.append(self._build_layout(view))

        # Lesson route: add theory card below workspace
        if is_active and view.lesson_panel is not None:
            theory_card = self._build_theory_card(view)
            if theory_card:
                content_panels.append(theory_card)

        return ft.View(
            route=view.route,
            controls=[
                self._build_top_bar(view),
                ft.Row(
                    controls=[
                        self._build_nav_rail(view.route),
                        ft.VerticalDivider(width=1),
                        ft.Container(
                            content=ft.Column(
                                controls=content_panels,
                                spacing=SpacingScale.SM,
                                scroll=ft.ScrollMode.AUTO,
                            ),
                            padding=SpacingScale.LG,
                            expand=True,
                        ),
                    ],
                    expand=True,
                ),
            ],
        )

    def _build_theory_card(self, view: RouteUIModel) -> Any:
        """Build the lesson theory card below the workspace."""
        if ft is None or view.lesson_panel is None:
            return None
        tokens = ThemeManager.get_tokens(self.page)

        sess = self.session
        obj_met = False
        if sess and sess.lesson_config and sess.lesson_progress:
            obj = sess.lesson_config.objective
            sim = sess.simulator
            obj_met = evaluate_lesson_objective(
                objective=obj,
                steps_executed=sim.state.current_step,
                cumulative_reward=sim.state.cumulative_reward,
                cumulative_regret=sim.state.cumulative_regret,
            )

        # Build stage stepper mini
        stage_dots: list[Any] = []
        for i in range(1, 6):
            is_cur = i == view.lesson_panel.stage_index
            is_done = i < view.lesson_panel.stage_index
            stage_dots.append(
                ft.Container(
                    content=ft.Text(
                        "✓" if is_done else str(i),
                        size=9,
                        weight=ft.FontWeight.BOLD,
                        color=tokens.text_on_accent if (is_cur or is_done) else tokens.text_muted,
                    ),
                    width=20,
                    height=20,
                    border_radius=10,
                    bgcolor=(
                        tokens.environment_accent
                        if is_done
                        else (tokens.environment_accent if is_cur else tokens.bg_tertiary)
                    ),
                    alignment=ft.alignment.center,
                )
            )
            if i < 5:
                stage_dots.append(
                    ft.Container(
                        content=ft.Text(">", size=9, color=tokens.text_muted),
                    )
                )

        next_button: list[Any] = []
        if obj_met and not view.lesson_panel.stage_index >= 5:

            def _advance(e: Any) -> None:
                s = self.session
                if s and s.lesson_progress:
                    progress = s.lesson_progress.advance()
                    if progress.current_stage > 5:
                        progress = progress.mark_completed()
                    s.lesson_progress = progress
                    self._refresh_view()

            next_button.append(
                ft.FilledTonalButton(
                    text="Next Stage →",
                    on_click=_advance,
                    icon=ft.Icons.ARROW_FORWARD,
                )
            )

        return ft.Container(
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            ft.Column(
                                controls=[
                                    ft.Text(
                                        f"STAGE {view.lesson_panel.stage_index} · {view.lesson_panel.lesson_title.upper()}",
                                        size=FontScale.SMALL,
                                        weight=ft.FontWeight.W_500,
                                        color=tokens.environment_accent,
                                    ),
                                    ft.Text(
                                        view.lesson_panel.objective_text,
                                        size=FontScale.SMALL,
                                        color=tokens.text_secondary,
                                    ),
                                ],
                                expand=True,
                            ),
                            ft.Row(controls=stage_dots, spacing=4, tight=True, visible=False),
                        ],
                        spacing=SpacingScale.SM,
                    ),
                    ft.Divider(height=1),
                    ft.Container(
                        content=ft.Markdown(
                            view.lesson_panel.theory_markdown,
                            selectable=True,
                        ),
                        bgcolor=tokens.bg_tertiary,
                        border_radius=6,
                        padding=SpacingScale.MD,
                    ),
                    ft.Row(
                        controls=[
                            ft.Text(
                                view.lesson_panel.step_explanation,
                                size=FontScale.SMALL,
                                italic=True,
                                color=tokens.text_muted,
                            ),
                            *next_button,
                        ],
                        spacing=SpacingScale.SM,
                    ),
                ],
                spacing=SpacingScale.SM,
                tight=True,
            ),
            border=ft.border.all(0.5, tokens.surface_border),
            border_radius=10,
            padding=SpacingScale.MD,
        )

    def _build_layout(self, view: RouteUIModel) -> Any:
        """Build the split-workspace layout for active routes."""
        if ft is None:
            return None

        env_controls, inter_controls, agent_controls = [], [], []

        # Environment zone
        if view.scene_panel:
            env_controls.append(
                ft.Text(
                    value=view.scene_panel.world_title,
                    size=FontScale.BODY,
                    weight=ft.FontWeight.W_600,
                )
            )
            env_controls.append(
                ft.Text(value=view.scene_panel.world_description, size=FontScale.SMALL)
            )
            for key, value in view.scene_panel.context_items.items():
                if key != "step":
                    env_controls.append(ft.Text(value=f"{key}: {value}", size=FontScale.SMALL))

        # Interaction zone
        if view.treatment_cards:
            inter_controls.append(
                ft.Text(
                    "Treatment Cards",
                    size=FontScale.SMALL,
                    weight=ft.FontWeight.W_600,
                )
            )
            for card in view.treatment_cards:
                suffix = "  ← selected" if card.selected else ""
                inter_controls.append(
                    ft.Text(
                        value=f"{card.label}{suffix}",
                        size=FontScale.BODY,
                        weight=ft.FontWeight.W_500 if card.selected else None,
                    )
                )

        # Agent zone
        if view.lesson_panel:
            agent_controls.append(
                ft.Text(
                    f"Lesson: {view.lesson_panel.lesson_title}",
                    size=FontScale.SMALL,
                    weight=ft.FontWeight.BOLD,
                )
            )
            agent_controls.append(
                ft.Text(f"Stage: {view.lesson_panel.stage_index}/5", size=FontScale.SMALL)
            )

        return SplitWorkspaceLayout.build(
            self.page,
            view,
            self.session,
            environment_controls=env_controls,
            interaction_controls=inter_controls,
            agent_controls=agent_controls,
            chart_controls=[],
        )

    # ── Route handling ───────────────────────────────────────────────

    def _on_route_change(self, event: Any) -> None:
        self._refresh_view()

    def _refresh_view(self) -> None:
        route = self.page.route or "/"
        sess = self.session
        if not sess:
            return
        try:
            view = build_route_ui_model(
                route,
                prefs=sess.prefs,
                trace_records=tuple(sess.simulator.trace_buffer.to_records()),
                sim_context=sess.simulator.world.sample_context(
                    sess.simulator.state.current_step + 1
                ),
                lesson_progress=sess.lesson_progress,
                lesson_config=sess.lesson_config,
                sim_step_index=sess.simulator.state.current_step,
                sim_cumulative_reward=sess.simulator.state.cumulative_reward,
                sim_cumulative_regret=sess.simulator.state.cumulative_regret,
            )
            self.page.views.clear()
            self.page.views.append(self._render_view(view))
            self.page.update()
        except Exception:
            import traceback

            traceback.print_exc()

    # ── Autoplay ─────────────────────────────────────────────────────

    def _start_autoplay(self) -> None:
        """Start the autoplay loop as an async task."""
        if self._autoplay_task is not None:
            return  # already running

        async def _autoplay_loop() -> None:
            while True:
                sess = self.session
                if not sess or sess._cancel_autoplay:
                    break
                sess.simulator.step()
                sess.controller.step()
                self._advance_lesson_if_ready()
                self._refresh_view()
                speed = sess.prefs.speed
                try:
                    multiplier = float(speed.replace("x", ""))
                except (ValueError, AttributeError):
                    multiplier = 1.0
                delay = max(0.02, 1.0 / multiplier * 0.5)
                await asyncio.sleep(delay)

        self._autoplay_task = asyncio.create_task(_autoplay_loop())

    def _stop_autoplay(self) -> None:
        """Stop the autoplay loop."""
        sess = self.session
        if sess:
            sess._cancel_autoplay = True
        if self._autoplay_task:
            self._autoplay_task.cancel()
            self._autoplay_task = None

    def _cancel_autoplay(self) -> None:
        self._stop_autoplay()

    def _advance_lesson_if_ready(self) -> bool:
        """Check whether lesson objectives are met and advance stage if so."""
        sess = self.session
        if not sess or not sess.lesson_config or not sess.lesson_progress:
            return False
        if sess.lesson_progress.completed:
            return False

        obj = sess.lesson_config.objective
        sim = sess.simulator
        if evaluate_lesson_objective(
            objective=obj,
            steps_executed=sim.state.current_step,
            cumulative_reward=sim.state.cumulative_reward,
            cumulative_regret=sim.state.cumulative_regret,
        ):
            progress = sess.lesson_progress.advance()
            if progress.current_stage > 5:
                progress = progress.mark_completed()
            sess.lesson_progress = progress
            return True
        return False

    # ── Theme ────────────────────────────────────────────────────────

    def apply_theme(self, mode: str = "light") -> None:
        ThemeManager.apply_theme(self.page, mode)
