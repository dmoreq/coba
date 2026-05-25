"""Consolidated view-model dataclasses for the web UI.

Merged from: scene_panel.py, treatment_card.py, trace_table.py,
batch_summary_panel.py, snapshot_diff_view.py, lesson_models.py,
context_inspection.py, tooltips.py, and comparison/snapshot_diff.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScenePanelModel:
    """View-model for scenario context rendering."""

    world_title: str
    world_description: str
    context_items: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TreatmentCardModel:
    """View-model for an action card."""

    arm_id: str
    label: str
    predicted_score: float | None = None
    selected: bool = False


@dataclass(frozen=True)
class TraceTableModel:
    """Pure data model for trace table rendering."""

    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    total_entries: int
    filter_query: str = ""


@dataclass(frozen=True)
class BatchSummaryPanelProps:
    """Props for rendering a batch summary statistics panel."""

    summaries: tuple[Any, ...]  # tuple of PolicySummaryStats
    sort_by: str = "mean_reward"
    sort_descending: bool = True


@dataclass(frozen=True)
class SnapshotDiffResult:
    """Diff output for UI rendering."""

    changed_keys: tuple[str, ...]
    before: dict[str, Any]
    after: dict[str, Any]


@dataclass(frozen=True)
class DiffViewProps:
    """Props for rendering a diff view from two snapshots."""

    title: str
    label_before: str
    label_after: str
    diff: SnapshotDiffResult


@dataclass(frozen=True)
class LessonPanelModel:
    """View-model for lesson theory/progression panel."""

    lesson_id: str
    lesson_title: str
    stage_index: int
    theory_markdown: str
    locked_controls: tuple[str, ...]
    objective_text: str
    step_explanation: str


@dataclass(frozen=True)
class ContextInspectionModel:
    """Feature vector and per-feature contribution preview."""

    feature_order: tuple[str, ...]
    feature_values: tuple[float, ...]
    notes: str


@dataclass(frozen=True)
class ParamTooltip:
    """Pedagogical tooltip payload for one control."""

    title: str
    intuition: str
    formula: str
    tuning_hint: str


def build_trace_table(
    records: list[dict[str, Any]],
    *,
    filter_query: str = "",
) -> TraceTableModel:
    """Build a TraceTableModel from raw trace records."""
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


def build_batch_summary_panel(
    summaries: list[Any],
    *,
    sort_by: str = "mean_reward",
    sort_descending: bool = True,
) -> BatchSummaryPanelProps:
    """Build a BatchSummaryPanelProps from a list of PolicySummaryStats."""
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


def build_diff_view_props(
    *,
    before: SnapshotDiffResult | None = None,
    after: SnapshotDiffResult | None = None,
    title: str = "",
    label_before: str = "",
    label_after: str = "",
) -> DiffViewProps:
    """Build DiffViewProps from optional before/after snapshots."""
    title_final = title or "Snapshot Diff"
    label_before_final = label_before or "Before"
    label_after_final = label_after or "After"
    if before is not None and after is not None:
        diff = SnapshotDiffResult(
            changed_keys=tuple(sorted(set(before.changed_keys) | set(after.changed_keys))),
            before=before.after,
            after=after.before,
        )
    elif before is not None:
        diff = before
    elif after is not None:
        diff = after
    else:
        diff = SnapshotDiffResult(changed_keys=(), before={}, after={})

    return DiffViewProps(
        title=title_final,
        label_before=label_before_final,
        label_after=label_after_final,
        diff=diff,
    )


__all__ = [
    "BatchSummaryPanelProps",
    "ContextInspectionModel",
    "DiffViewProps",
    "LessonPanelModel",
    "ParamTooltip",
    "ScenePanelModel",
    "SnapshotDiffResult",
    "TraceTableModel",
    "TreatmentCardModel",
    "build_batch_summary_panel",
    "build_diff_view_props",
    "build_trace_table",
]
