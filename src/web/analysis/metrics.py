"""Arena chart metric builders from trace records."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SeriesPoint:
    """One x/y chart point."""

    step: int
    value: float


@dataclass(frozen=True)
class ArenaMetrics:
    """Computed arena metrics for chart/table rendering."""

    reward_series: tuple[SeriesPoint, ...]
    regret_series: tuple[SeriesPoint, ...]
    arm_pull_counts: dict[str, int]
    uncertainty_series: tuple[SeriesPoint, ...]


def build_arena_metrics(records: list[dict[str, Any]]) -> ArenaMetrics:
    """Build core arena chart metrics from serialized trace records."""
    reward_series = tuple(
        SeriesPoint(step=int(record["step_index"]), value=float(record["cumulative_reward"]))
        for record in records
    )
    regret_series = tuple(
        SeriesPoint(step=int(record["step_index"]), value=float(record["cumulative_regret"]))
        for record in records
    )
    pull_counts = Counter(str(record["chosen_arm"]) for record in records)

    uncertainty_series = tuple(
        SeriesPoint(
            step=int(record["step_index"]),
            value=float(record.get("metadata", {}).get("uncertainty", 0.0)),
        )
        for record in records
    )
    return ArenaMetrics(
        reward_series=reward_series,
        regret_series=regret_series,
        arm_pull_counts=dict(pull_counts),
        uncertainty_series=uncertainty_series,
    )
