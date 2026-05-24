"""Configurable continuous world backed by WorldConfig."""

from __future__ import annotations

import math
import random
from typing import Any

from web.continuous.simulator import ContinuousWorld
from web.worlds.schema import FeatureDef, WorldConfig


class ConfigurableContinuousWorld(ContinuousWorld):
    """Continuous-action world from WorldConfig with quadratic reward landscape."""

    def __init__(self, config: WorldConfig, action_scale: float = 0.2) -> None:
        self.config = config
        self.action_scale = action_scale
        self._rng = random.Random(0)
        self._feature_defs = {f.name: f for f in config.features}
        self._arms = list(config.arms)

    def reset(self, seed: int | None = None) -> None:
        self._rng = random.Random(0 if seed is None else seed)

    def sample_context(self, step_index: int) -> dict[str, Any]:
        context: dict[str, Any] = {"step": step_index}
        for feature in self.config.features:
            context[feature.name] = self._sample_feature(feature)
        return context

    def sample_reward(self, context: dict[str, Any], action: float) -> float:
        optimal = self._compute_optimal_action(context)
        distance = (action - optimal) / self.action_scale
        prob = max(0.0, 1.0 - distance * distance)
        prob = min(1.0, prob)
        return 1.0 if self._rng.random() < prob else 0.0

    def compute_optimal_action(self, context: dict[str, Any]) -> float:
        return self._compute_optimal_action(context)

    def _compute_optimal_action(self, context: dict[str, Any]) -> float:
        """Derive optimal action as weighted average of arm base rates."""
        total_weight = 0.0
        weighted_sum = 0.0
        for arm in self._arms:
            prob = self._expected_probability(context, arm)
            weighted_sum += prob * arm.base_rate
            total_weight += abs(prob) + 1e-6
        if total_weight < 1e-9:
            return 0.5
        return weighted_sum / total_weight

    def _sample_feature(self, feature: FeatureDef) -> Any:
        if feature.feature_type == "numeric":
            return self._rng.uniform(feature.numeric_min, feature.numeric_max)
        if feature.feature_type == "binary":
            return 1 if self._rng.random() < 0.5 else 0
        if feature.feature_type == "categorical":
            return self._rng.choice(list(feature.categories))
        raise ValueError(f"Unsupported feature type: {feature.feature_type}")

    def _expected_probability(self, context: dict[str, Any], arm: Any) -> float:
        base = min(1.0 - 1e-6, max(1e-6, arm.base_rate))
        score = math.log(base / (1.0 - base))
        for feature_name, weight in arm.weights.items():
            feature_def = self._feature_defs[feature_name]
            score += weight * self._numeric_feature_value(feature_def, context[feature_name])
        return 1.0 / (1.0 + math.exp(-score))

    def _numeric_feature_value(self, feature: FeatureDef, value: Any) -> float:
        if feature.feature_type == "numeric":
            denom = feature.numeric_max - feature.numeric_min
            if denom == 0:
                return 0.0
            return max(0.0, min(1.0, (float(value) - feature.numeric_min) / denom))
        if feature.feature_type == "binary":
            return float(value)
        if feature.feature_type == "categorical":
            cats = list(feature.categories)
            if len(cats) == 1:
                return 0.0
            return cats.index(str(value)) / float(len(cats) - 1)
        raise ValueError(f"Unsupported feature type: {feature.feature_type}")
