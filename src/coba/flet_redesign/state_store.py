"""Application state store for route/world/policy selections."""

from __future__ import annotations

from dataclasses import dataclass

from coba.flet_redesign.worlds import ConfigurableWorld, create_world, list_world_configs


@dataclass
class AppSelectionState:
    """Current high-level app selections."""

    world_id: str
    policy_id: str = "random"


class AppStateStore:
    """Minimal state-store with world switching for Phase 2."""

    def __init__(self, default_world_id: str | None = None) -> None:
        available = list_world_configs()
        if not available:
            raise ValueError("No worlds registered")
        self._state = AppSelectionState(world_id=default_world_id or available[0].world_id)

    @property
    def state(self) -> AppSelectionState:
        return self._state

    def switch_world(self, world_id: str) -> AppSelectionState:
        create_world(world_id)  # validate id via registry
        self._state = AppSelectionState(world_id=world_id, policy_id=self._state.policy_id)
        return self._state

    def build_world(self) -> ConfigurableWorld:
        return create_world(self._state.world_id)
