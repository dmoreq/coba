"""Split Workspace Dashboard — 3-zone horizontal layout with bottom charts.

Layout structure:
  TOP ROW: Environment (left) | Interaction (center) | Agent (right)
  BOTTOM: Charts zone (collapsible)
"""

from __future__ import annotations

from typing import Any

from web.theme import ColorTokens, FontScale, SpacingScale
from web.theme.theme_manager import ThemeManager

try:
    import flet as ft
except ModuleNotFoundError:  # pragma: no cover
    ft = None  # type: ignore[assignment]


class SplitWorkspaceLayout:
    """Builds the Option C split-workspace dashboard layout."""

    @staticmethod
    def _build_zone(
        page: Any,
        tokens: ColorTokens,
        title: str,
        accent_color: str,
        bg_color: str,
        controls: list[Any],
        expand: int = 1,
    ) -> Any:
        if ft is None:
            return None
        return ft.Container(
            expand=expand,
            padding=SpacingScale.MD,
            border=ft.Border(
                left=ft.BorderSide(1, tokens.surface_border),
                top=ft.BorderSide(1, tokens.surface_border),
                right=ft.BorderSide(1, tokens.surface_border),
                bottom=ft.BorderSide(1, tokens.surface_border),
            ),
            border_radius=8,
            bgcolor=bg_color,
            content=ft.Column(
                controls=[
                    ft.Text(
                        value=title,
                        size=FontScale.SMALL,
                        weight=ft.FontWeight.BOLD,
                        color=accent_color,
                    ),
                    ft.Container(height=2, bgcolor=accent_color, border_radius=1),
                    ft.Divider(height=1),
                    *controls,
                ],
                spacing=SpacingScale.XS,
                scroll=ft.ScrollMode.AUTO,
                tight=True,
            ),
        )

    @staticmethod
    def build(
        page: Any,
        view_model: Any,
        session: Any,
        *,
        environment_controls: list[Any] | None = None,
        interaction_controls: list[Any] | None = None,
        agent_controls: list[Any] | None = None,
        chart_controls: list[Any] | None = None,
    ) -> Any:
        """Build the split workspace layout.

        Args:
            page: Flet page.
            view_model: RouteUIModel or similar.
            session: _SimSession or AppState.
            environment_controls: Controls for the left (environment) zone.
            interaction_controls: Controls for the center (interaction) zone.
            agent_controls: Controls for the right (agent) zone.
            chart_controls: Controls for the bottom charts zone.
        """
        if ft is None:
            return None

        tokens = ThemeManager.get_tokens(page)
        env = environment_controls or []
        inter = interaction_controls or []
        agent = agent_controls or []
        charts = chart_controls or []

        top_row = ft.Row(
            controls=[
                SplitWorkspaceLayout._build_zone(
                    page,
                    tokens,
                    "Environment",
                    tokens.environment_accent,
                    tokens.environment_zone_bg,
                    env,
                    expand=1,
                ),
                SplitWorkspaceLayout._build_zone(
                    page,
                    tokens,
                    "Interaction",
                    tokens.text_muted,
                    tokens.interaction_zone_bg,
                    inter,
                    expand=2,
                ),
                SplitWorkspaceLayout._build_zone(
                    page,
                    tokens,
                    "Agent",
                    tokens.agent_accent,
                    tokens.agent_zone_bg,
                    agent,
                    expand=1,
                ),
            ],
            spacing=SpacingScale.SM,
            expand=True,
        )

        charts_expandable = ft.ExpansionTile(
            title=ft.Text("Charts", size=FontScale.SMALL, weight=ft.FontWeight.BOLD),
            initially_expanded=len(charts) > 0,
            collapsed_icon=ft.Icons.EXPAND_MORE,
            expanded_icon=ft.Icons.EXPAND_LESS,
            controls=[
                ft.Row(controls=charts, spacing=SpacingScale.SM, expand=True),
            ],
        )

        return ft.Column(
            controls=[top_row, charts_expandable],
            spacing=SpacingScale.SM,
            expand=True,
        )
