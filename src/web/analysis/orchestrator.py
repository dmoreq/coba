"""Run orchestrators for side-by-side policy comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from web.policy_factory import build_policy
from web.simulator import DiscreteSimulator
from web.state import RunConfig
from web.worlds import create_world, get_world_config


@dataclass(frozen=True)
class ComparisonRunResult:
    """One policy run outcome in comparison mode."""

    policy_id: str
    world_id: str
    seed: int
    horizon: int
    cumulative_reward: float
    cumulative_regret: float
    trace_records: list[dict[str, Any]]


def run_policy_comparison(
    *,
    world_id: str,
    policy_ids: list[str],
    seed: int,
    horizon: int,
) -> list[ComparisonRunResult]:
    """Run multiple policies on the same world/seed settings."""
    config = get_world_config(world_id)
    feature_order = tuple(feature.name for feature in config.features)
    results: list[ComparisonRunResult] = []
    for policy_id in policy_ids:
        world = create_world(world_id)
        policy = build_policy(policy_id, feature_order=feature_order, seed=seed)
        simulator = DiscreteSimulator(
            policy=policy,
            world=world,
            config=RunConfig(seed=seed, horizon=horizon),
        )
        simulator.reset()
        simulator.run_steps(horizon)
        results.append(
            ComparisonRunResult(
                policy_id=policy_id,
                world_id=world_id,
                seed=seed,
                horizon=horizon,
                cumulative_reward=simulator.state.cumulative_reward,
                cumulative_regret=simulator.state.cumulative_regret,
                trace_records=simulator.trace_buffer.to_records(),
            )
        )
    return results


def run_batch_comparison(
    *,
    world_id: str,
    policy_ids: list[str],
    seeds: list[int],
    horizon: int,
) -> list[ComparisonRunResult]:
    """Run comparison for multiple seeds and flatten results."""
    all_results: list[ComparisonRunResult] = []
    for seed in seeds:
        all_results.extend(
            run_policy_comparison(
                world_id=world_id,
                policy_ids=policy_ids,
                seed=seed,
                horizon=horizon,
            )
        )
    return all_results
