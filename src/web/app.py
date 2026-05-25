"""AppShell — root application shell for the COBA web app.

Handles page structure, navigation, theme, and route dispatch.
"""

from __future__ import annotations

import asyncio
from typing import Any

from web.components.theme_toggle import build_theme_toggle
from web.layouts.split_workspace import SplitWorkspaceLayout
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

    def __init__(self, page: Any, session: Any, pref_store: Any) -> None:
        self.page = page
        self.session = session
        self.pref_store = pref_store
        self._autoplay_task: asyncio.Task[None] | None = None

        # Wire navigation
        page.on_route_change = self._on_route_change

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

        if self.session.lesson_progress:
            actions.insert(
                0,
                ft.Text(
                    f"Stage {self.session.lesson_progress.current_stage}/5",
                    italic=True,
                    size=FontScale.SMALL,
                ),
            )

        return ft.AppBar(
            title=ft.Text(f"COBA · {view.title}", size=FontScale.TITLE),
            actions=actions,
            bgcolor=(
                self.page.theme.color_scheme.primary
                if hasattr(self.page.theme, "color_scheme")
                else None
            ),
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
                ft.Text(
                    value=view.scene_panel.world_description,
                    size=FontScale.SMALL,
                )
            )
            for key, value in view.scene_panel.context_items.items():
                if key != "step":
                    env_controls.append(
                        ft.Text(
                            value=f"{key}: {value}",
                            size=FontScale.SMALL,
                        )
                    )

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
                ft.Text(
                    f"Stage: {view.lesson_panel.stage_index}/5",
                    size=FontScale.SMALL,
                )
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
        try:
            view = build_route_ui_model(
                route,
                prefs=self.session.prefs,
                trace_records=tuple(self.session.simulator.trace_buffer.to_records()),
                sim_context=self.session.simulator.world.sample_context(
                    self.session.simulator.state.current_step + 1
                ),
                lesson_progress=self.session.lesson_progress,
                lesson_config=self.session.lesson_config,
                sim_step_index=self.session.simulator.state.current_step,
                sim_cumulative_reward=self.session.simulator.state.cumulative_reward,
                sim_cumulative_regret=self.session.simulator.state.cumulative_regret,
            )
            self.page.views.clear()
            self.page.views.append(self._render_view(view))
            self.page.update()
        except Exception:
            import traceback

            traceback.print_exc()

    # ── Autoplay ─────────────────────────────────────────────────────

    def _cancel_autoplay(self) -> None:
        if self._autoplay_task:
            self._autoplay_task.cancel()
            self._autoplay_task = None

    # ── Theme ────────────────────────────────────────────────────────

    def apply_theme(self, mode: str = "light") -> None:
        ThemeManager.apply_theme(self.page, mode)
