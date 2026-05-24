"""State containers for simulation execution and replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from web.contracts import SimulationStepResult


@dataclass
class RunConfig:
    """Deterministic run configuration."""

    seed: int = 0
    horizon: int = 100
    autoplay_interval_ms: int = 300


@dataclass
class ArmState:
    """Per-arm aggregates for charts and summaries."""

    arm: Any
    pulls: int = 0
    reward_sum: float = 0.0

    @property
    def mean_reward(self) -> float:
        if self.pulls == 0:
            return 0.0
        return self.reward_sum / float(self.pulls)


@dataclass
class SimulationState:
    """Mutable run state captured by the app state store."""

    config: RunConfig = field(default_factory=RunConfig)
    current_step: int = 0
    cumulative_reward: float = 0.0
    cumulative_regret: float = 0.0
    is_running: bool = False
    trace: list[SimulationStepResult] = field(default_factory=list)

    def append_step(self, step: SimulationStepResult) -> None:
        """Update aggregates and append one step into trace history."""
        self.current_step = step.step_index
        self.cumulative_reward = step.cumulative_reward
        self.cumulative_regret = step.cumulative_regret
        self.trace.append(step)

    def reset(self) -> None:
        """Reset state for a new run while preserving run config."""
        self.current_step = 0
        self.cumulative_reward = 0.0
        self.cumulative_regret = 0.0
        self.is_running = False
        self.trace.clear()
