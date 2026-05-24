"""Snapshot diff comparison component."""

from __future__ import annotations

from dataclasses import dataclass

from web.comparison.snapshot_diff import SnapshotDiffResult


@dataclass(frozen=True)
class DiffViewProps:
    """Props for rendering a diff view from two snapshots."""

    title: str
    label_before: str
    label_after: str
    diff: SnapshotDiffResult


def build_diff_view_props(
    *,
    before: SnapshotDiffResult | None = None,
    after: SnapshotDiffResult | None = None,
    title: str = "",
    label_before: str = "",
    label_after: str = "",
) -> DiffViewProps:
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
