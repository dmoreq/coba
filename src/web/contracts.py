"""Core contracts for the Flet redesign execution model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

ArmT = TypeVar("ArmT")
ContextT = TypeVar("ContextT")
ContextTContra = TypeVar("ContextTContra", contravariant=True)


@dataclass(frozen=True)
class SimulationStepResult:
    """Immutable payload emitted after each simulation step."""

    step_index: int
    context: Any
    chosen_arm: Any
    reward: float
    cumulative_reward: float
    cumulative_regret: float
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DebugSnapshotProvider(Protocol):
    """Optional contract for objects that can expose debug internals."""

    def get_debug_snapshot(self) -> dict[str, Any]:
        """Return serializable internals for debugger rendering."""


@runtime_checkable
class BanditPolicy(Protocol[ArmT, ContextTContra]):
    """Policy contract used by the simulation loop."""

    def reset(self) -> None:
        """Reset policy state for a new run."""

    def select_arm(self, context: ContextTContra, arms: Sequence[ArmT]) -> ArmT:
        """Choose one arm for the provided context."""

    def update(self, context: ContextTContra, arm: ArmT, reward: float) -> None:
        """Update policy state after observing a reward."""


@runtime_checkable
class World(Protocol[ArmT, ContextT]):
    """World contract used to generate contexts and rewards."""

    def reset(self, seed: int | None = None) -> None:
        """Reset world state to a reproducible starting point."""

    def get_available_arms(self) -> Sequence[ArmT]:
        """Return currently available arms."""

    def sample_context(self, step_index: int) -> ContextT:
        """Return context for one simulation step."""

    def sample_reward(self, context: ContextT, arm: ArmT) -> float:
        """Return sampled reward for one selected arm."""


@runtime_checkable
class Simulator(Protocol):
    """Unified simulator contract shared by discrete and continuous loops."""

    def reset(self) -> None:
        """Reset simulator to initial state."""

    def step(self) -> Any:
        """Advance one step and return the emitted result."""

    def run_steps(self, n_steps: int) -> list[Any]:
        """Advance n steps and return emitted results."""

    def replay_payload(self) -> dict[str, Any]:
        """Return deterministic replay payload for export/verification."""
