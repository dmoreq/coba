"""Run control state machine for step/play/pause/reset interactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RunMode = Literal["idle", "running", "paused"]


@dataclass(frozen=True)
class RunControlState:
    """Current run-controller state."""

    mode: RunMode = "idle"
    steps_executed: int = 0


class RunController:
    """Deterministic run-control transitions used by UI callbacks."""

    def __init__(self) -> None:
        self._state = RunControlState()

    @property
    def state(self) -> RunControlState:
        return self._state

    def play(self) -> RunControlState:
        self._state = RunControlState(mode="running", steps_executed=self._state.steps_executed)
        return self._state

    def pause(self) -> RunControlState:
        self._state = RunControlState(mode="paused", steps_executed=self._state.steps_executed)
        return self._state

    def step(self) -> RunControlState:
        next_mode: RunMode = "paused" if self._state.mode == "running" else self._state.mode
        self._state = RunControlState(mode=next_mode, steps_executed=self._state.steps_executed + 1)
        return self._state

    def reset(self) -> RunControlState:
        self._state = RunControlState(mode="idle", steps_executed=0)
        return self._state
