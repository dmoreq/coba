"""Performance smoke baseline for redesign simulation loop."""

from __future__ import annotations

import random
import time
from collections.abc import Sequence

from coba.flet_redesign.contracts import World
from coba.flet_redesign.policy_factory import build_policy
from coba.flet_redesign.simulator import DiscreteSimulator
from coba.flet_redesign.state import RunConfig


class PerfWorld(World[str, dict[str, float]]):
    def __init__(self) -> None:
        self._rng = random.Random(0)
        self._arms = ("a", "b", "c")

    def reset(self, seed: int | None = None) -> None:
        self._rng = random.Random(0 if seed is None else seed)

    def get_available_arms(self) -> Sequence[str]:
        return self._arms

    def sample_context(self, step_index: int) -> dict[str, float]:
        return {
            "x1": float(step_index % 10),
            "x2": float(step_index % 3),
            "x3": 1.0,
        }

    def sample_reward(self, context: dict[str, float], arm: str) -> float:
        _ = context
        probs = {"a": 0.4, "b": 0.6, "c": 0.5}
        return 1.0 if self._rng.random() < probs[arm] else 0.0


def test_advanced_policy_step_rate_smoke() -> None:
    feature_order = ("x1", "x2", "x3")
    policy_ids = ["gp_ucb", "bootstrapped_ensemble", "linucb_hybrid", "tree_ucb", "tree_ts"]
    start = time.perf_counter()
    for policy_id in policy_ids:
        policy = build_policy(policy_id, feature_order=feature_order, seed=3)
        sim = DiscreteSimulator(
            policy=policy,
            world=PerfWorld(),
            config=RunConfig(seed=3, horizon=150),
        )
        sim.reset()
        sim.run_steps(150)
    elapsed = time.perf_counter() - start
    assert elapsed < 8.0
