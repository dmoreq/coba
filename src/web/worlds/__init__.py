"""World schemas, configurations, and factories for Flet redesign."""

from web.worlds.base import ConfigurableWorld
from web.worlds.core_worlds import (
    CORE_WORLD_CONFIGS,
    MOVIEMATCH_WORLD,
    NEWSFEED_WORLD,
    RURAL_CLINIC_WORLD,
)
from web.worlds.presets import (
    CONTEXTUAL_PRESETS,
    ContextualPreset,
    list_contextual_presets,
)
from web.worlds.registry import create_world, get_world_config, list_world_configs
from web.worlds.schema import ArmDef, FeatureDef, WorldConfig

__all__ = [
    "ArmDef",
    "ConfigurableWorld",
    "CONTEXTUAL_PRESETS",
    "CORE_WORLD_CONFIGS",
    "ContextualPreset",
    "FeatureDef",
    "MOVIEMATCH_WORLD",
    "NEWSFEED_WORLD",
    "RURAL_CLINIC_WORLD",
    "WorldConfig",
    "create_world",
    "get_world_config",
    "list_world_configs",
    "list_contextual_presets",
]
