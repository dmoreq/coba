"""Comparison orchestration and statistics utilities."""

from web.comparison.orchestrator import (
    ComparisonRunResult,
    run_batch_comparison,
    run_policy_comparison,
)
from web.comparison.snapshot_diff import (
    SnapshotDiffResult,
    diff_debug_snapshots,
    diff_trace_records,
)
from web.comparison.stats import PolicySummaryStats, summarize_comparison_runs

__all__ = [
    "ComparisonRunResult",
    "PolicySummaryStats",
    "SnapshotDiffResult",
    "diff_debug_snapshots",
    "diff_trace_records",
    "run_batch_comparison",
    "run_policy_comparison",
    "summarize_comparison_runs",
]
