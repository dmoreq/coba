"""Analysis package — merged from arena/ and comparison/ modules.

Provides metric computation, run comparison orchestration, diagnostics, and statistics.
"""

from web.analysis.comparison import ArenaRunStore, RunSnapshot
from web.analysis.diagnostics import ComparisonDiagnostics, compute_comparison_diagnostics
from web.analysis.metrics import ArenaMetrics, SeriesPoint, build_arena_metrics
from web.analysis.orchestrator import (
    ComparisonRunResult,
    run_batch_comparison,
    run_policy_comparison,
)
from web.analysis.stats import PolicySummaryStats, summarize_comparison_runs

__all__ = [
    "ArenaMetrics",
    "ArenaRunStore",
    "ComparisonDiagnostics",
    "ComparisonRunResult",
    "PolicySummaryStats",
    "RunSnapshot",
    "SeriesPoint",
    "build_arena_metrics",
    "compute_comparison_diagnostics",
    "run_batch_comparison",
    "run_policy_comparison",
    "summarize_comparison_runs",
]
