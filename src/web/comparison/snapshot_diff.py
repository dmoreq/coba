"""Trace/debug snapshot diff helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SnapshotDiffResult:
    """Diff output for UI rendering."""

    changed_keys: tuple[str, ...]
    before: dict[str, Any]
    after: dict[str, Any]


def diff_trace_records(
    before_records: list[dict[str, Any]],
    after_records: list[dict[str, Any]],
) -> SnapshotDiffResult:
    before_last = before_records[-1] if before_records else {}
    after_last = after_records[-1] if after_records else {}
    return _diff_dicts(before_last, after_last)


def diff_debug_snapshots(before: dict[str, Any], after: dict[str, Any]) -> SnapshotDiffResult:
    return _diff_dicts(before, after)


def _diff_dicts(before: dict[str, Any], after: dict[str, Any]) -> SnapshotDiffResult:
    keys = sorted(set(before.keys()) | set(after.keys()))
    changed = [
        key
        for key in keys
        if json.dumps(before.get(key), sort_keys=True, default=str)
        != json.dumps(after.get(key), sort_keys=True, default=str)
    ]
    return SnapshotDiffResult(changed_keys=tuple(changed), before=before, after=after)
