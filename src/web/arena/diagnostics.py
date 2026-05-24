"""Comparative diagnostics for advanced policy runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ComparisonDiagnostics:
    """Derived diagnostics between two run traces."""

    adaptation_lag_steps: int
    mean_uncertainty: float
    final_reward_delta: float


def compute_comparison_diagnostics(
    *,
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
) -> ComparisonDiagnostics:
    """Compute lightweight diagnostics used by comparison views."""
    baseline_lag = _adaptation_lag(baseline_records)
    candidate_lag = _adaptation_lag(candidate_records)
    adaptation_lag = max(0, baseline_lag - candidate_lag)

    candidate_uncertainty = [
        float(record.get("metadata", {}).get("uncertainty", 0.0)) for record in candidate_records
    ]
    mean_uncertainty = (
        sum(candidate_uncertainty) / float(len(candidate_uncertainty))
        if candidate_uncertainty
        else 0.0
    )

    baseline_final = float(baseline_records[-1]["cumulative_reward"]) if baseline_records else 0.0
    candidate_final = (
        float(candidate_records[-1]["cumulative_reward"]) if candidate_records else 0.0
    )
    return ComparisonDiagnostics(
        adaptation_lag_steps=adaptation_lag,
        mean_uncertainty=mean_uncertainty,
        final_reward_delta=candidate_final - baseline_final,
    )


def _adaptation_lag(records: list[dict[str, Any]]) -> int:
    if not records:
        return 0
    final_reward = float(records[-1]["cumulative_reward"])
    threshold = 0.8 * final_reward
    for record in records:
        if float(record["cumulative_reward"]) >= threshold:
            return int(record["step_index"])
    return int(records[-1]["step_index"])
