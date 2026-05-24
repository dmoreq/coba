"""Batch summary statistics panel for multi-seed comparison results."""

from __future__ import annotations

from dataclasses import dataclass

from web.comparison.stats import PolicySummaryStats


@dataclass(frozen=True)
class BatchSummaryPanelProps:
    """Props for rendering a batch summary statistics panel."""

    summaries: tuple[PolicySummaryStats, ...]
    sort_by: str = "mean_reward"
    sort_descending: bool = True


def build_batch_summary_panel(
    summaries: list[PolicySummaryStats],
    *,
    sort_by: str = "mean_reward",
    sort_descending: bool = True,
) -> BatchSummaryPanelProps:
    if sort_descending:
        summaries_sorted = sorted(
            summaries,
            key=lambda s: getattr(s, sort_by, 0.0),
            reverse=True,
        )
    else:
        summaries_sorted = sorted(
            summaries,
            key=lambda s: getattr(s, sort_by, 0.0),
        )
    return BatchSummaryPanelProps(
        summaries=tuple(summaries_sorted),
        sort_by=sort_by,
        sort_descending=sort_descending,
    )
