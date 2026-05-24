"""Config-driven world implementation."""

from __future__ import annotations

import math
import random
from typing import Any

from coba.flet_redesign.contracts import World
from coba.flet_redesign.worlds.schema import ArmDef, FeatureDef, WorldConfig


class ConfigurableWorld(World[str, dict[str, Any]]):
    """World implementation backed by :class:`WorldConfig`."""

    def __init__(self, config: WorldConfig) -> None:
        self.config = config
        self._rng = random.Random(0)
        self._feature_defs = {feature.name: feature for feature in config.features}
        self._arms = list(config.arms)

    def reset(self, seed: int | None = None) -> None:
        self._rng = random.Random(0 if seed is None else seed)

    def get_available_arms(self) -> tuple[str, ...]:
        return tuple(arm.arm_id for arm in self._arms)

    def sample_context(self, step_index: int) -> dict[str, Any]:
        context: dict[str, Any] = {"step": step_index}
        for feature in self.config.features:
            context[feature.name] = self._sample_feature(feature)
        return context

    def sample_reward(self, context: dict[str, Any], arm: str) -> float:
        arm_def = self._arm_for(arm)
        prob = self._expected_probability(context=context, arm=arm_def)
        return 1.0 if self._rng.random() < prob else 0.0

    def expected_rewards(self, context: dict[str, Any]) -> dict[str, float]:
        """Return expected reward probabilities for all available arms."""
        return {
            arm.arm_id: self._expected_probability(context=context, arm=arm) for arm in self._arms
        }

    def _sample_feature(self, feature: FeatureDef) -> Any:
        if feature.feature_type == "numeric":
            return self._rng.uniform(feature.numeric_min, feature.numeric_max)
        if feature.feature_type == "binary":
            return 1 if self._rng.random() < 0.5 else 0
        if feature.feature_type == "categorical":
            return self._rng.choice(list(feature.categories))
        raise ValueError(f"Unsupported feature type: {feature.feature_type}")

    def _arm_for(self, arm_id: str) -> ArmDef:
        for arm in self._arms:
            if arm.arm_id == arm_id:
                return arm
        raise ValueError(f"Unknown arm '{arm_id}'")

    def _expected_probability(self, context: dict[str, Any], arm: ArmDef) -> float:
        # Convert base rate to logit, apply weighted feature delta, then sigmoid.
        base = min(1.0 - 1e-6, max(1e-6, arm.base_rate))
        score = math.log(base / (1.0 - base))
        for feature_name, weight in arm.weights.items():
            feature_def = self._feature_defs[feature_name]
            score += weight * self._numeric_feature_value(feature_def, context[feature_name])
        return 1.0 / (1.0 + math.exp(-score))

    def _numeric_feature_value(self, feature: FeatureDef, value: Any) -> float:
        if feature.feature_type == "numeric":
            denominator = feature.numeric_max - feature.numeric_min
            if denominator == 0:
                return 0.0
            scaled = (float(value) - feature.numeric_min) / denominator
            return max(0.0, min(1.0, scaled))
        if feature.feature_type == "binary":
            return float(value)
        if feature.feature_type == "categorical":
            categories = list(feature.categories)
            if len(categories) == 1:
                return 0.0
            index = categories.index(str(value))
            return index / float(len(categories) - 1)
        raise ValueError(f"Unsupported feature type: {feature.feature_type}")
