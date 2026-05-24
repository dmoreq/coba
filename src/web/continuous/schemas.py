"""Schemas for continuous action simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ContinuousActionSpace:
    """Continuous action bounds."""

    min_value: float
    max_value: float

    def __post_init__(self) -> None:
        if self.min_value >= self.max_value:
            raise ValueError("min_value must be < max_value")

    def clip(self, value: float) -> float:
        return max(self.min_value, min(self.max_value, value))


@dataclass(frozen=True)
class ContinuousStepResult:
    """One continuous-action simulation step."""

    step_index: int
    context: dict[str, Any]
    action: float
    reward: float
    cumulative_reward: float
    metadata: dict[str, Any] = field(default_factory=dict)
