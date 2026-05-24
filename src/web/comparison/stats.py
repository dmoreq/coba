"""Summary statistics for comparison runs."""

from __future__ import annotations

import math
from dataclasses import dataclass

from web.comparison.orchestrator import ComparisonRunResult


@dataclass(frozen=True)
class PolicySummaryStats:
    """Aggregate metrics for one policy over repeated seeds."""

    policy_id: str
    n_runs: int
    mean_reward: float
    std_reward: float
    ci95_half_width: float
    mean_regret: float


def summarize_comparison_runs(results: list[ComparisonRunResult]) -> list[PolicySummaryStats]:
    grouped: dict[str, list[ComparisonRunResult]] = {}
    for result in results:
        grouped.setdefault(result.policy_id, []).append(result)

    summaries: list[PolicySummaryStats] = []
    for policy_id, rows in grouped.items():
        rewards = [row.cumulative_reward for row in rows]
        regrets = [row.cumulative_regret for row in rows]
        n_runs = len(rows)
        mean_reward = sum(rewards) / float(n_runs)
        mean_regret = sum(regrets) / float(n_runs)
        if n_runs > 1:
            variance = sum((reward - mean_reward) ** 2 for reward in rewards) / float(n_runs - 1)
            std_reward = math.sqrt(max(0.0, variance))
            ci95 = 1.96 * std_reward / math.sqrt(float(n_runs))
        else:
            std_reward = 0.0
            ci95 = 0.0
        summaries.append(
            PolicySummaryStats(
                policy_id=policy_id,
                n_runs=n_runs,
                mean_reward=mean_reward,
                std_reward=std_reward,
                ci95_half_width=ci95,
                mean_regret=mean_regret,
            )
        )

    return sorted(summaries, key=lambda row: row.mean_reward, reverse=True)
