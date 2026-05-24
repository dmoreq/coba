"""Phase-1 deterministic replay and regret regression tests."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from coba.flet_redesign.contracts import World
from coba.flet_redesign.policies import (
    EpsilonGreedyPolicy,
    RandomPolicy,
    SoftmaxPolicy,
    ThompsonSamplingPolicy,
    UCB1Policy,
)
from coba.flet_redesign.simulator import DiscreteSimulator
from coba.flet_redesign.state import RunConfig


@dataclass
class BernoulliBanditWorld(World[str, dict[str, int]]):
    """Two-arm Bernoulli world with deterministic seeding on reset."""

    probs: dict[str, float]

    def __post_init__(self) -> None:
        self._rng = random.Random(0)

    def reset(self, seed: int | None = None) -> None:
        self._rng = random.Random(0 if seed is None else seed)

    def get_available_arms(self) -> Sequence[str]:
        return tuple(self.probs.keys())

    def sample_context(self, step_index: int) -> dict[str, int]:
        return {"step": step_index}

    def sample_reward(self, context: dict[str, int], arm: str) -> float:
        _ = context
        return 1.0 if self._rng.random() < self.probs[arm] else 0.0


def _optimal_reward(_: Any, __: list[str]) -> float:
    return 0.8


def _run(policy: Any, seed: int, horizon: int = 400) -> tuple[float, float]:
    world = BernoulliBanditWorld(probs={"bad": 0.2, "good": 0.8})
    sim = DiscreteSimulator(
        policy=policy,
        world=world,
        config=RunConfig(seed=seed, horizon=horizon),
        optimal_reward_fn=_optimal_reward,
    )
    sim.reset()
    sim.run_steps(horizon)
    return sim.state.cumulative_reward, sim.state.cumulative_regret


def test_replay_payload_is_deterministic_for_same_seed() -> None:
    horizon = 120
    policy1 = UCB1Policy(alpha=1.0, seed=99)
    policy2 = UCB1Policy(alpha=1.0, seed=99)
    world1 = BernoulliBanditWorld(probs={"bad": 0.2, "good": 0.8})
    world2 = BernoulliBanditWorld(probs={"bad": 0.2, "good": 0.8})

    sim1 = DiscreteSimulator(
        policy=policy1,
        world=world1,
        config=RunConfig(seed=17, horizon=horizon),
        optimal_reward_fn=_optimal_reward,
    )
    sim2 = DiscreteSimulator(
        policy=policy2,
        world=world2,
        config=RunConfig(seed=17, horizon=horizon),
        optimal_reward_fn=_optimal_reward,
    )

    sim1.reset()
    sim2.reset()
    sim1.run_steps(horizon)
    sim2.run_steps(horizon)

    assert sim1.replay_payload() == sim2.replay_payload()


def test_replay_payload_changes_for_different_seed() -> None:
    horizon = 120
    policy = UCB1Policy(alpha=1.0, seed=99)
    world = BernoulliBanditWorld(probs={"bad": 0.2, "good": 0.8})
    sim_a = DiscreteSimulator(
        policy=policy,
        world=world,
        config=RunConfig(seed=17, horizon=horizon),
        optimal_reward_fn=_optimal_reward,
    )
    sim_a.reset()
    sim_a.run_steps(horizon)
    payload_a = sim_a.replay_payload()

    policy_b = UCB1Policy(alpha=1.0, seed=99)
    world_b = BernoulliBanditWorld(probs={"bad": 0.2, "good": 0.8})
    sim_b = DiscreteSimulator(
        policy=policy_b,
        world=world_b,
        config=RunConfig(seed=18, horizon=horizon),
        optimal_reward_fn=_optimal_reward,
    )
    sim_b.reset()
    sim_b.run_steps(horizon)
    payload_b = sim_b.replay_payload()

    assert payload_a != payload_b


def test_regret_baseline_advanced_policies_outperform_random() -> None:
    seeds = [2, 7, 11, 17, 23]
    horizon = 400

    def average_reward(factory: Callable[[int], Any]) -> float:
        rewards = [_run(factory(seed), seed=seed, horizon=horizon)[0] for seed in seeds]
        return sum(rewards) / float(len(rewards))

    random_reward = average_reward(lambda seed: RandomPolicy(seed=seed))
    epsilon_reward = average_reward(lambda seed: EpsilonGreedyPolicy(epsilon=0.1, seed=seed))
    ucb_reward = average_reward(lambda seed: UCB1Policy(alpha=1.0, seed=seed))
    thompson_reward = average_reward(lambda seed: ThompsonSamplingPolicy(seed=seed))
    softmax_reward = average_reward(lambda seed: SoftmaxPolicy(tau=0.2, seed=seed))

    assert epsilon_reward > random_reward + 15.0
    assert softmax_reward > random_reward + 8.0
    assert ucb_reward > random_reward + 30.0
    assert thompson_reward > random_reward + 30.0
