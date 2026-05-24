"""Trace table rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TraceTableModel:
    """Pure data model for trace table rendering."""

    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    total_entries: int
    filter_query: str = ""


def build_trace_table(
    records: list[dict[str, Any]],
    *,
    filter_query: str = "",
) -> TraceTableModel:
    if filter_query.strip():
        from web.trace import filter_trace_records

        records = filter_trace_records(records, filter_query)

    if not records:
        return TraceTableModel(
            columns=(
                "step_index",
                "chosen_arm",
                "reward",
                "cumulative_reward",
                "cumulative_regret",
            ),
            rows=(),
            total_entries=0,
            filter_query=filter_query,
        )

    columns = ("step_index", "chosen_arm", "reward", "cum_reward", "cum_regret")
    rows: list[tuple[str, ...]] = []
    for r in records[-200:]:  # cap at 200 rows for rendering
        rows.append(
            (
                str(r.get("step_index", "")),
                str(r.get("chosen_arm", "")),
                f"{float(r.get('reward', 0.0)):.3f}",
                f"{float(r.get('cumulative_reward', 0.0)):.3f}",
                f"{float(r.get('cumulative_regret', 0.0)):.3f}",
            )
        )
    return TraceTableModel(
        columns=columns,
        rows=tuple(rows),
        total_entries=len(records),
        filter_query=filter_query,
    )
