"""World registry and construction helpers."""

from __future__ import annotations

from coba.flet_redesign.worlds.base import ConfigurableWorld
from coba.flet_redesign.worlds.core_worlds import CORE_WORLD_CONFIGS
from coba.flet_redesign.worlds.schema import WorldConfig

WORLD_CONFIG_REGISTRY: dict[str, WorldConfig] = {
    config.world_id: config for config in CORE_WORLD_CONFIGS
}


def list_world_configs() -> tuple[WorldConfig, ...]:
    """List registered worlds in stable order."""
    return tuple(CORE_WORLD_CONFIGS)


def get_world_config(world_id: str) -> WorldConfig:
    """Get one world configuration by id."""
    try:
        return WORLD_CONFIG_REGISTRY[world_id]
    except KeyError as exc:
        raise KeyError(f"Unknown world_id '{world_id}'") from exc


def create_world(world_id: str) -> ConfigurableWorld:
    """Build a configurable world instance from registry config."""
    return ConfigurableWorld(config=get_world_config(world_id))
