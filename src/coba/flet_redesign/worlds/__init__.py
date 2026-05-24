"""World schemas, configurations, and factories for Flet redesign."""

from coba.flet_redesign.worlds.base import ConfigurableWorld
from coba.flet_redesign.worlds.core_worlds import (
    CORE_WORLD_CONFIGS,
    MOVIEMATCH_WORLD,
    NEWSFEED_WORLD,
    RURAL_CLINIC_WORLD,
)
from coba.flet_redesign.worlds.presets import (
    CONTEXTUAL_PRESETS,
    ContextualPreset,
    list_contextual_presets,
)
from coba.flet_redesign.worlds.registry import create_world, get_world_config, list_world_configs
from coba.flet_redesign.worlds.schema import ArmDef, FeatureDef, WorldConfig

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
