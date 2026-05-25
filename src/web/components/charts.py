"""Chart rendering components — cumulative regret, arm histogram, reward timeline.

Flet 0.85.1 does not include ft.LineChart/ft.BarChart. Charts are built
from styled ft.Container bars and text-based sparklines instead.
"""

from __future__ import annotations

from typing import Any

from web.theme import FontScale, SpacingScale
from web.theme.theme_manager import ThemeManager

try:
    import flet as ft
except ModuleNotFoundError:  # pragma: no cover
    ft = None  # type: ignore[assignment]


def build_regret_chart(
    page: Any,
    points: list[tuple[int, float]] | None = None,
    max_points: int = 100,
) -> Any:
    """Build a cumulative regret display using a horizontal bar chart of containers."""
    if ft is None:
        return None
    tokens = ThemeManager.get_tokens(page)
    points = points or []

    if not points:
        return ft.Text(
            value="No regret data yet", size=FontScale.SMALL, color=tokens.text_muted, italic=True
        )

    displayed = points[-max_points:]
    max_val = max(v for _, v in displayed) if displayed else 1.0

    bars: list[Any] = []
    for step, value in displayed[-20:]:  # show last 20 as mini-bars
        pct = value / max_val if max_val > 0 else 0
        bars.append(
            ft.Row(
                controls=[
                    ft.Text(value=str(step), size=8, color=tokens.text_muted, width=30),
                    ft.Container(
                        width=int(120 * pct),
                        height=8,
                        bgcolor=tokens.chart_line_primary,
                        border_radius=1,
                    ),
                    ft.Text(value=f"{value:.2f}", size=8, color=tokens.text_primary, width=50),
                ],
                spacing=SpacingScale.XS,
                tight=True,
            )
        )

    last = points[-1][1] if points else 0
    summary = ft.Text(
        value=f"Cumulative Regret: {last:.3f} (after {len(points)} steps)",
        size=FontScale.SMALL,
        weight=ft.FontWeight.W_600,
        color=tokens.chart_line_primary,
    )

    return ft.Container(
        content=ft.Column(controls=[summary] + bars, spacing=2, tight=True),
        border=ft.border.only(bottom=ft.BorderSide(1, tokens.chart_grid)),
        padding=SpacingScale.SM,
    )


def build_arm_histogram(
    page: Any,
    labels: list[str] | None = None,
    values: list[int] | None = None,
) -> Any:
    """Build an arm selection histogram from colored containers."""
    if ft is None:
        return None
    tokens = ThemeManager.get_tokens(page)
    labels = labels or []
    values = values or []

    if not labels:
        return ft.Text(
            value="No arm data yet", size=FontScale.SMALL, color=tokens.text_muted, italic=True
        )

    max_count = max(values) if values else 0
    if max_count == 0:
        return ft.Text(
            value="No pulls recorded yet",
            size=FontScale.SMALL,
            color=tokens.text_muted,
            italic=True,
        )

    bars: list[Any] = []
    for label, count in zip(labels, values):
        pct = count / max_count if max_count > 0 else 0
        tooltip = f"{label}: {count} pulls"
        bars.append(
            ft.Row(
                controls=[
                    ft.Text(
                        value=label, size=FontScale.CAPTION, color=tokens.text_secondary, width=80
                    ),
                    ft.Container(
                        width=int(140 * pct),
                        height=14,
                        bgcolor=tokens.agent_accent,
                        border_radius=2,
                        tooltip=tooltip,
                    ),
                    ft.Text(
                        value=str(count), size=FontScale.CAPTION, color=tokens.text_muted, width=30
                    ),
                ],
                spacing=SpacingScale.XS,
                tight=True,
            )
        )

    total = sum(values)
    summary = ft.Text(
        value=f"Arm Pulls (total: {total})",
        size=FontScale.SMALL,
        weight=ft.FontWeight.W_600,
        color=tokens.agent_accent,
    )

    return ft.Container(
        content=ft.Column(controls=[summary] + bars, spacing=SpacingScale.XS, tight=True),
        border=ft.border.only(bottom=ft.BorderSide(1, tokens.chart_grid)),
        padding=SpacingScale.SM,
    )


def build_reward_timeline(
    page: Any,
    rewards: list[float] | None = None,
    max_points: int = 20,
) -> Any:
    """Build a compact reward sparkline from colored dot indicators."""
    if ft is None:
        return None
    tokens = ThemeManager.get_tokens(page)
    rewards = rewards or []

    if not rewards:
        return ft.Text(
            value="No rewards yet", size=FontScale.SMALL, color=tokens.text_muted, italic=True
        )

    displayed = rewards[-max_points:]
    dots: list[Any] = []
    for r in displayed:
        is_success = r > 0
        dots.append(
            ft.Container(
                width=8,
                height=8,
                border_radius=4,
                bgcolor=tokens.success_feedback if is_success else tokens.regret_feedback,
                tooltip=f"Reward: {r:.1f}",
            )
        )

    success_count = sum(1 for r in displayed if r > 0)
    summary = ft.Text(
        value=f"Recent: {success_count}/{len(displayed)} successes",
        size=FontScale.CAPTION,
        color=tokens.success_feedback,
    )

    return ft.Container(
        content=ft.Column(
            controls=[
                summary,
                ft.Row(controls=dots, spacing=2, tight=True),
            ],
            spacing=SpacingScale.XS,
            tight=True,
        ),
        padding=SpacingScale.SM,
    )
