"""Side-by-side policy comparison page."""

from __future__ import annotations

from dataclasses import dataclass

from web.comparison.orchestrator import ComparisonRunResult, run_policy_comparison
from web.comparison.stats import PolicySummaryStats, summarize_comparison_runs
from web.worlds import list_world_configs

ALL_POLICY_IDS: tuple[str, ...] = (
    "random",
    "epsilon_greedy",
    "ucb1",
    "thompson",
    "softmax",
    "linucb",
    "linucb_sw",
    "logistic_ucb",
    "gp_ucb",
    "bootstrapped_ensemble",
    "linucb_hybrid",
    "tree_ucb",
    "tree_ts",
    "cats",
)

POLICY_DISPLAY: dict[str, str] = {
    "random": "Random",
    "epsilon_greedy": "Epsilon-Greedy",
    "ucb1": "UCB1",
    "thompson": "Thompson Sampling",
    "softmax": "Softmax",
    "linucb": "LinUCB",
    "linucb_sw": "LinUCB-SW",
    "logistic_ucb": "Logistic UCB",
    "gp_ucb": "GP-UCB",
    "bootstrapped_ensemble": "Bootstrap Ensemble",
    "linucb_hybrid": "LinUCB Hybrid",
    "tree_ucb": "Tree UCB",
    "tree_ts": "Tree TS",
    "cats": "CATS",
}


@dataclass(frozen=True)
class ComparisonPageModel:
    """Pure view-model for the comparison page."""

    world_id: str
    selected_policy_ids: tuple[str, ...]
    seed: int
    horizon: int
    results: tuple[ComparisonRunResult, ...]
    summary_stats: tuple[PolicySummaryStats, ...]
    available_worlds: tuple[str, ...]
    available_policies: tuple[str, ...]
    error: str = ""


def build_comparison_model(
    *,
    world_id: str = "rural_clinic",
    policy_ids: list[str] | None = None,
    seed: int = 0,
    horizon: int = 200,
    run_comparison: bool = False,
) -> ComparisonPageModel:
    worlds = tuple(w.world_id for w in list_world_configs())
    selected = tuple(policy_ids) if policy_ids else ("random", "epsilon_greedy", "ucb1")
    available_policies = tuple(pid for pid in ALL_POLICY_IDS if pid not in selected)

    if not run_comparison:
        return ComparisonPageModel(
            world_id=world_id,
            selected_policy_ids=selected,
            seed=seed,
            horizon=horizon,
            results=(),
            summary_stats=(),
            available_worlds=worlds,
            available_policies=available_policies,
        )

    try:
        results_list = run_policy_comparison(
            world_id=world_id,
            policy_ids=list(selected),
            seed=seed,
            horizon=horizon,
        )
        summaries_list = summarize_comparison_runs(results_list)
        return ComparisonPageModel(
            world_id=world_id,
            selected_policy_ids=selected,
            seed=seed,
            horizon=horizon,
            results=tuple(results_list),
            summary_stats=tuple(summaries_list),
            available_worlds=worlds,
            available_policies=available_policies,
        )
    except Exception as exc:
        return ComparisonPageModel(
            world_id=world_id,
            selected_policy_ids=selected,
            seed=seed,
            horizon=horizon,
            results=(),
            summary_stats=(),
            available_worlds=worlds,
            available_policies=available_policies,
            error=str(exc),
        )
