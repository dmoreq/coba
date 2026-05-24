"""World configuration schema and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

FeatureType = Literal["numeric", "categorical", "binary"]


@dataclass(frozen=True)
class FeatureDef:
    """Feature definition used by world context generation."""

    name: str
    feature_type: FeatureType
    numeric_min: float = 0.0
    numeric_max: float = 1.0
    categories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FeatureDef.name cannot be empty")
        if self.feature_type == "numeric" and self.numeric_min >= self.numeric_max:
            raise ValueError("numeric_min must be < numeric_max for numeric features")
        if self.feature_type == "categorical" and len(self.categories) < 2:
            raise ValueError("categorical features require at least two categories")


@dataclass(frozen=True)
class ArmDef:
    """Arm definition with a base reward and linear feature weights."""

    arm_id: str
    label: str
    base_rate: float
    weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.arm_id:
            raise ValueError("ArmDef.arm_id cannot be empty")
        if not (0.0 <= self.base_rate <= 1.0):
            raise ValueError("ArmDef.base_rate must be in [0, 1]")


@dataclass(frozen=True)
class WorldConfig:
    """Top-level world configuration."""

    world_id: str
    title: str
    description: str
    difficulty: str
    features: tuple[FeatureDef, ...]
    arms: tuple[ArmDef, ...]

    def __post_init__(self) -> None:
        if not self.world_id:
            raise ValueError("WorldConfig.world_id cannot be empty")
        if len(self.features) == 0:
            raise ValueError("WorldConfig.features cannot be empty")
        if len(self.arms) < 2:
            raise ValueError("WorldConfig.arms requires at least two arms")

        feature_names = [feature.name for feature in self.features]
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("WorldConfig feature names must be unique")

        arm_ids = [arm.arm_id for arm in self.arms]
        if len(arm_ids) != len(set(arm_ids)):
            raise ValueError("WorldConfig arm ids must be unique")

        valid_features = set(feature_names)
        for arm in self.arms:
            unknown = set(arm.weights.keys()) - valid_features
            if unknown:
                missing = ", ".join(sorted(unknown))
                raise ValueError(f"Arm '{arm.arm_id}' has unknown feature weights: {missing}")
