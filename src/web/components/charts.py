"""Chart rendering components — cumulative regret, arm histogram, reward timeline."""

from __future__ import annotations

from typing import Any

from web.theme.theme_manager import ThemeManager

try:
    import flet as ft
except ModuleNotFoundError:  # pragma: no cover
    ft = None  # type: ignore[assignment]


def _get_chart_data(points: list[tuple[int, float]]) -> list[Any]:
    """Convert (step, value) tuples to Flet LineChartDataPoint objects."""
    return [ft.LineChartDataPoint(x=s, y=v) for s, v in points]


def build_regret_chart(
    page: Any,
    points: list[tuple[int, float]] | None = None,
    max_points: int = 100,
) -> Any:
    """Build a cumulative regret LineChart."""
    if ft is None:
        return None
    tokens = ThemeManager.get_tokens(page)
    points = points or []

    series = ft.LineChartData(
        data_points=_get_chart_data(points[-max_points:]),
        color=tokens.chart_line_primary,
        stroke_width=2,
        curved=True,
        prevent_curve_over_shooting=True,
    )

    return ft.LineChart(
        data_series=[series],
        border=ft.Border(bottom=ft.BorderSide(1, tokens.chart_grid)),
        left_axis=ft.ChartAxis(
            labels_size=30,
            labels=(
                [
                    ft.ChartAxisLabel(value=v, label=ft.Text(str(int(v)), size=9))
                    for v in range(0, int(max([v for _, v in points] or [10])))
                    if points
                ]
                if points
                else None
            ),
        ),
        bottom_axis=ft.ChartAxis(labels_size=20),
        tooltip_bgcolor=tokens.bg_tertiary,
        bgcolor=tokens.chart_bg,
        min_y=0,
        animate=True,
        animation_duration=300,
        expand=True,
    )


def build_arm_histogram(
    page: Any,
    labels: list[str] | None = None,
    values: list[int] | None = None,
) -> Any:
    """Build an arm selection histogram BarChart."""
    if ft is None:
        return None
    tokens = ThemeManager.get_tokens(page)
    labels = labels or []
    values = values or []

    groups = [
        ft.BarChartGroup(
            x=i,
            bar_roster=[
                ft.BarChartRod(
                    from_y=0,
                    to_y=values[i] if i < len(values) else 0,
                    color=tokens.agent_accent,
                    tooltip=f"{labels[i]}: {values[i]} pulls" if i < len(labels) else "",
                )
            ],
        )
        for i in range(len(labels))
    ]

    return ft.BarChart(
        bar_groups=groups,
        border=ft.Border(bottom=ft.BorderSide(1, tokens.chart_grid)),
        left_axis=ft.ChartAxis(labels_size=20),
        bottom_axis=ft.ChartAxis(
            labels=[
                ft.ChartAxisLabel(value=i, label=ft.Text(labels[i], size=9))
                for i in range(len(labels))
            ],
        ),
        bgcolor=tokens.chart_bg,
        animate=True,
        animation_duration=300,
        expand=True,
    )


def build_reward_timeline(
    page: Any,
    rewards: list[float] | None = None,
    max_points: int = 20,
) -> Any:
    """Build a compact reward sparkline."""
    if ft is None:
        return None
    tokens = ThemeManager.get_tokens(page)
    rewards = rewards or []

    points = [ft.LineChartDataPoint(x=i, y=r) for i, r in enumerate(rewards[-max_points:])]
    points_list = list(points)

    series = ft.LineChartData(
        data_points=points_list,
        color=tokens.success_feedback,
        stroke_width=1.5,
        curved=False,
    )

    return ft.LineChart(
        data_series=[series],
        height=60,
        border=ft.Border(bottom=ft.BorderSide(1, tokens.chart_grid)),
        bgcolor=tokens.chart_bg,
        min_y=0,
        max_y=1,
        animate=True,
        animation_duration=300,
        expand=True,
    )
