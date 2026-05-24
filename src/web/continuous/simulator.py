"""Continuous-action simulation loop with regret and replay."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from typing import Protocol

from web.continuous.cats_policy import CATSLikePolicy
from web.continuous.schemas import ContinuousStepResult


class ContinuousWorld(Protocol):
    """World contract for continuous-action simulation."""

    def reset(self, seed: int | None = None) -> None:
        """Reset world state."""

    def sample_context(self, step_index: int) -> dict[str, Any]:
        """Sample context for current step."""

    def sample_reward(self, context: dict[str, Any], action: float) -> float:
        """Return reward for continuous action."""


class ContinuousSimulator:
    """Simulator for continuous action policies with regret tracking and replay."""

    def __init__(
        self,
        policy: CATSLikePolicy,
        world: ContinuousWorld,
        seed: int = 0,
        optimal_reward_fn: Callable[[dict[str, Any]], float] | None = None,
    ) -> None:
        self.policy = policy
        self.world = world
        self.seed = seed
        self.current_step = 0
        self.cumulative_reward = 0.0
        self.cumulative_regret = 0.0
        self.trace: list[ContinuousStepResult] = []
        self._optimal_reward_fn = optimal_reward_fn

    def reset(self) -> None:
        self.policy.reset()
        self.world.reset(seed=self.seed)
        self.current_step = 0
        self.cumulative_reward = 0.0
        self.cumulative_regret = 0.0
        self.trace.clear()

    def step(self) -> ContinuousStepResult:
        step_index = self.current_step + 1
        context = self.world.sample_context(step_index)
        action = self.policy.select_action()
        reward = float(self.world.sample_reward(context, action))

        optimal_reward = reward
        if self._optimal_reward_fn is not None:
            optimal_reward = float(self._optimal_reward_fn(context))
        regret = max(0.0, optimal_reward - reward)

        self.policy.update(action, reward)
        self.cumulative_reward += reward
        self.cumulative_regret += regret
        self.current_step = step_index

        result = ContinuousStepResult(
            step_index=step_index,
            context=context,
            action=action,
            reward=reward,
            cumulative_reward=self.cumulative_reward,
            metadata={
                "debug": self.policy.get_debug_snapshot(),
                "regret": regret,
                "cumulative_regret": self.cumulative_regret,
            },
        )
        self.trace.append(result)
        return result

    def run_steps(self, n_steps: int) -> list[ContinuousStepResult]:
        if n_steps < 0:
            raise ValueError("n_steps must be >= 0")
        return [self.step() for _ in range(n_steps)]

    def replay_payload(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "steps": [
                {
                    "step_index": r.step_index,
                    "action": r.action,
                    "reward": r.reward,
                    "cumulative_reward": r.cumulative_reward,
                    "cumulative_regret": r.metadata.get("cumulative_regret", 0.0),
                }
                for r in self.trace
            ],
        }
