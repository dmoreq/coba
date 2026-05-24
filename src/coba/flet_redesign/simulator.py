"""Discrete simulation loop for Flet redesign scaffolding."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from coba.flet_redesign.contracts import BanditPolicy, SimulationStepResult, World
from coba.flet_redesign.state import RunConfig, SimulationState
from coba.flet_redesign.trace import TraceBuffer


class DiscreteSimulator:
    """Minimal deterministic-compatible simulator for discrete arm policies."""

    def __init__(
        self,
        policy: BanditPolicy[Any, Any],
        world: World[Any, Any],
        config: RunConfig | None = None,
        optimal_reward_fn: Callable[[Any, list[Any]], float] | None = None,
    ) -> None:
        self.policy = policy
        self.world = world
        self.config = config or RunConfig()
        self.state = SimulationState(config=self.config)
        self.trace_buffer = TraceBuffer()
        self._optimal_reward_fn = optimal_reward_fn

    def reset(self) -> None:
        """Reset policy, world, and run state using configured seed."""
        self.policy.reset()
        self.world.reset(seed=self.config.seed)
        self.state.reset()
        self.trace_buffer.clear()

    def step(self) -> SimulationStepResult:
        """Advance one simulation step and return the emitted snapshot."""
        next_step = self.state.current_step + 1
        context = self.world.sample_context(next_step)
        arms = list(self.world.get_available_arms())
        chosen_arm = self.policy.select_arm(context=context, arms=arms)
        reward = float(self.world.sample_reward(context=context, arm=chosen_arm))
        self.policy.update(context=context, arm=chosen_arm, reward=reward)

        optimal_reward = reward
        if self._optimal_reward_fn is not None:
            optimal_reward = float(self._optimal_reward_fn(context, arms))
        step_regret = max(0.0, optimal_reward - reward)

        step_result = SimulationStepResult(
            step_index=next_step,
            context=context,
            chosen_arm=chosen_arm,
            reward=reward,
            cumulative_reward=self.state.cumulative_reward + reward,
            cumulative_regret=self.state.cumulative_regret + step_regret,
        )
        self.state.append_step(step_result)
        self.trace_buffer.append(step_result)
        return step_result
