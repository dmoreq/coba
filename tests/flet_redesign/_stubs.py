"""Shared stubs and mock objects for web module testing.

Consolidates duplicated stubs from across the test suite.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any


class DummyWorld:
    """Deterministic world with fixed arms and constant reward."""

    def __init__(self, arms: Sequence[str] | None = None, reward: float = 0.0) -> None:
        self._arms = list(arms) if arms else ["arm_a", "arm_b"]
        self._reward = reward

    def reset(self, seed: int | None = None) -> None:
        pass

    def get_available_arms(self) -> list[str]:
        return list(self._arms)

    def sample_context(self, step_index: int) -> dict[str, int]:
        return {"step": step_index}

    def sample_reward(self, context: Any, arm: str) -> float:
        _ = context, arm
        return self._reward


class GreedyStubPolicy:
    """Policy that always picks the first arm."""

    def __init__(self, arm_id: str = "arm_a") -> None:
        self.arm_id = arm_id
        self.reset_called = False

    def reset(self) -> None:
        self.reset_called = True

    def select_arm(self, context: Any, arms: Sequence[Any]) -> Any:
        return arms[0] if arms else self.arm_id

    def update(self, context: Any, arm: Any, reward: float) -> None:
        pass


class AlwaysBadPolicy:
    """Policy that picks the worst arm by reversing the list."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._calls: list[str] = []

    def reset(self) -> None:
        self._calls.clear()

    def select_arm(self, context: Any, arms: Sequence[Any]) -> Any:
        result = list(arms)[-1] if arms else None
        self._calls.append(str(result))
        return result

    def update(self, context: Any, arm: Any, reward: float) -> None:
        pass


class BernoulliBanditWorld:
    """World with fixed Bernoulli probabilities per arm."""

    def __init__(self, probs: dict[str, float]) -> None:
        if any(p < 0 or p > 1 for p in probs.values()):
            raise ValueError("probabilities must be in [0, 1]")
        self._probs = dict(probs)
        self._rng = random.Random(0)

    def reset(self, seed: int | None = None) -> None:
        self._rng = random.Random(0 if seed is None else seed)

    def get_available_arms(self) -> list[str]:
        return list(self._probs)

    def sample_context(self, step_index: int) -> dict[str, int]:
        return {"step": step_index}

    def sample_reward(self, context: Any, arm: str) -> float:
        _ = context
        return 1.0 if self._rng.random() < self._probs[arm] else 0.0

    def optimal_arm(self) -> str:
        return max(self._probs, key=self._probs.__getitem__)


class DummyDebugger:
    """Stub DebugSnapshotProvider."""

    def __init__(self, snapshot: dict[str, Any] | None = None) -> None:
        self._snapshot = snapshot or {"debug": True}

    def get_debug_snapshot(self) -> dict[str, Any]:
        return dict(self._snapshot)


class NotAProtocol:
    """Class that does not implement any protocol — used for negative tests."""

    pass
