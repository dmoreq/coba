"""Chart rendering utilities for arena and comparison views."""

from __future__ import annotations

from dataclasses import dataclass

from web.analysis import ArenaMetrics


@dataclass(frozen=True)
class ChartData:
    """Processed chart data ready for Flet rendering."""

    reward_points: tuple[tuple[int, float], ...]
    regret_points: tuple[tuple[int, float], ...]
    arm_pull_labels: tuple[str, ...]
    arm_pull_values: tuple[int, ...]
    uncertainty_points: tuple[tuple[int, float], ...]


def build_chart_data(metrics: ArenaMetrics) -> ChartData:
    reward_points = tuple((p.step, p.value) for p in metrics.reward_series)
    regret_points = tuple((p.step, p.value) for p in metrics.regret_series)
    arm_pull_labels = tuple(metrics.arm_pull_counts.keys())
    arm_pull_values = tuple(metrics.arm_pull_counts.values())
    uncertainty_points = tuple((p.step, p.value) for p in metrics.uncertainty_series)
    return ChartData(
        reward_points=reward_points,
        regret_points=regret_points,
        arm_pull_labels=arm_pull_labels,
        arm_pull_values=arm_pull_values,
        uncertainty_points=uncertainty_points,
    )
