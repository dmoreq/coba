"""Sandbox scenario editor and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from web.worlds import WorldConfig, get_world_config
from web.worlds.schema import ArmDef


@dataclass(frozen=True)
class SandboxScenario:
    """Custom scenario payload for sandbox runs."""

    world_id: str
    policy_id: str
    horizon: int
    params: dict[str, Any]


class SandboxEditor:
    """Mutable sandbox editor with basic validations."""

    def __init__(self, world_id: str, policy_id: str, horizon: int = 200) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be > 0")
        self._scenario = SandboxScenario(
            world_id=world_id,
            policy_id=policy_id,
            horizon=horizon,
            params={},
        )

    @property
    def scenario(self) -> SandboxScenario:
        return self._scenario

    def set_param(self, key: str, value: Any) -> SandboxScenario:
        updated = dict(self._scenario.params)
        updated[key] = value
        self._scenario = SandboxScenario(
            world_id=self._scenario.world_id,
            policy_id=self._scenario.policy_id,
            horizon=self._scenario.horizon,
            params=updated,
        )
        return self._scenario

    def build_world_override(self, arm_base_rates: dict[str, float]) -> WorldConfig:
        """Return world config with overridden arm base rates."""
        config = get_world_config(self._scenario.world_id)
        updated_arms: list[ArmDef] = []
        for arm in config.arms:
            if arm.arm_id in arm_base_rates:
                rate = float(arm_base_rates[arm.arm_id])
                updated_arms.append(
                    ArmDef(
                        arm_id=arm.arm_id,
                        label=arm.label,
                        base_rate=rate,
                        weights=arm.weights,
                    )
                )
            else:
                updated_arms.append(arm)
        return WorldConfig(
            world_id=config.world_id,
            title=config.title,
            description=config.description,
            difficulty=config.difficulty,
            features=config.features,
            arms=tuple(updated_arms),
        )
