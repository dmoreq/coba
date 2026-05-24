"""Tests for discrete simulator scaffolding."""

from __future__ import annotations

from collections.abc import Sequence

from coba.flet_redesign.contracts import BanditPolicy, World
from coba.flet_redesign.simulator import DiscreteSimulator
from coba.flet_redesign.state import RunConfig


class GreedyStubPolicy(BanditPolicy[str, dict[str, int]]):
    def __init__(self) -> None:
        self.updates: list[tuple[str, float]] = []

    def reset(self) -> None:
        self.updates.clear()

    def select_arm(self, context: dict[str, int], arms: Sequence[str]) -> str:
        _ = context
        return arms[0]

    def update(self, context: dict[str, int], arm: str, reward: float) -> None:
        _ = context
        self.updates.append((arm, reward))


class DummyWorld(World[str, dict[str, int]]):
    def __init__(self) -> None:
        self._arms = ["a", "b"]
        self.seed_used: int | None = None

    def reset(self, seed: int | None = None) -> None:
        self.seed_used = seed

    def get_available_arms(self) -> list[str]:
        return self._arms

    def sample_context(self, step_index: int) -> dict[str, int]:
        return {"step": step_index}

    def sample_reward(self, context: dict[str, int], arm: str) -> float:
        _ = context
        return 1.0 if arm == "a" else 0.2


def optimal_reward_fn(context: dict[str, int], arms: list[str]) -> float:
    _ = context, arms
    return 1.0


def test_simulator_step_updates_state_and_trace() -> None:
    simulator = DiscreteSimulator(
        policy=GreedyStubPolicy(),
        world=DummyWorld(),
        config=RunConfig(seed=11, horizon=10),
        optimal_reward_fn=optimal_reward_fn,
    )
    simulator.reset()
    step = simulator.step()
    assert step.step_index == 1
    assert step.chosen_arm == "a"
    assert step.cumulative_reward == 1.0
    assert step.cumulative_regret == 0.0
    assert simulator.state.current_step == 1
    assert len(simulator.trace_buffer.steps) == 1


def test_simulator_reset_resets_world_policy_and_state() -> None:
    policy = GreedyStubPolicy()
    world = DummyWorld()
    simulator = DiscreteSimulator(
        policy=policy,
        world=world,
        config=RunConfig(seed=5, horizon=10),
        optimal_reward_fn=optimal_reward_fn,
    )
    simulator.step()
    assert simulator.state.current_step == 1
    simulator.reset()
    assert simulator.state.current_step == 0
    assert simulator.trace_buffer.steps == []
    assert world.seed_used == 5
    assert policy.updates == []


def test_simulator_regret_accumulates_when_suboptimal_arm_selected() -> None:
    class AlwaysBadPolicy(GreedyStubPolicy):
        def select_arm(self, context: dict[str, int], arms: Sequence[str]) -> str:
            _ = context
            return arms[1]

    simulator = DiscreteSimulator(
        policy=AlwaysBadPolicy(),
        world=DummyWorld(),
        config=RunConfig(seed=0, horizon=10),
        optimal_reward_fn=optimal_reward_fn,
    )
    result = simulator.step()
    assert result.reward == 0.2
    assert result.cumulative_regret == 0.8
