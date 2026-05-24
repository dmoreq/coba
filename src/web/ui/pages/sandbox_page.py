"""Sandbox scenario editor page."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from web.sandbox import SandboxEditor, SandboxScenario
from web.worlds import get_world_config, list_world_configs


@dataclass(frozen=True)
class SandboxPageModel:
    """Pure view-model for the sandbox editor page."""

    world_id: str
    policy_id: str
    horizon: int
    params: dict[str, Any]
    arm_overrides: dict[str, float]
    available_worlds: tuple[str, ...]
    available_policies: tuple[str, ...]
    scenario: SandboxScenario | None = None
    result: dict[str, Any] = field(default_factory=dict)
    validation_error: str = ""
    run_error: str = ""


_ALL_POLICY_IDS = (
    "random",
    "epsilon_greedy",
    "ucb1",
    "thompson",
    "softmax",
    "linucb",
    "linucb_sw",
    "logistic_ucb",
    "gp_ucb",
    "bootstrapped_ensemble",
    "linucb_hybrid",
    "tree_ucb",
    "tree_ts",
    "cats",
)


def build_sandbox_model(
    *,
    world_id: str = "rural_clinic",
    policy_id: str = "random",
    horizon: int = 200,
    params: dict[str, Any] | None = None,
    arm_overrides: dict[str, float] | None = None,
    validate_only: bool = False,
    run_scenario: bool = False,
) -> SandboxPageModel:
    worlds = tuple(w.world_id for w in list_world_configs())
    available_policies = tuple(pid for pid in _ALL_POLICY_IDS if pid != policy_id)
    params_final = params or {}
    arm_overrides_final = arm_overrides or {}

    editor = SandboxEditor(world_id=world_id, policy_id=policy_id, horizon=horizon)
    scenario = editor.scenario
    for key, value in params_final.items():
        scenario = editor.set_param(key, value)

    if validate_only and arm_overrides_final:
        try:
            editor.build_world_override(arm_overrides_final)
        except Exception as exc:
            return SandboxPageModel(
                world_id=world_id,
                policy_id=policy_id,
                horizon=horizon,
                params=params_final,
                arm_overrides=arm_overrides_final,
                available_worlds=worlds,
                available_policies=available_policies,
                scenario=scenario,
                validation_error=str(exc),
            )

    if run_scenario:
        from web.policy_factory import build_policy
        from web.simulator import DiscreteSimulator
        from web.state import RunConfig
        from web.worlds import ConfigurableWorld, create_world

        try:
            if arm_overrides_final:
                override_config = editor.build_world_override(arm_overrides_final)
                world: Any = ConfigurableWorld(config=override_config)
            else:
                world = create_world(world_id)

            config = get_world_config(world_id)
            feature_order = tuple(f.name for f in config.features)
            policy = build_policy(policy_id, feature_order=feature_order, seed=0)
            sim = DiscreteSimulator(
                policy=policy,
                world=world,
                config=RunConfig(seed=0, horizon=horizon),
            )
            sim.reset()
            sim.run_steps(horizon)
            result = {
                "steps": sim.state.current_step,
                "cumulative_reward": sim.state.cumulative_reward,
                "cumulative_regret": sim.state.cumulative_regret,
                "chosen_arm_last": (
                    sim.trace_buffer.to_records()[-1]["chosen_arm"]
                    if sim.trace_buffer.to_records()
                    else "n/a"
                ),
            }
            return SandboxPageModel(
                world_id=world_id,
                policy_id=policy_id,
                horizon=horizon,
                params=params_final,
                arm_overrides=arm_overrides_final,
                available_worlds=worlds,
                available_policies=available_policies,
                scenario=scenario,
                result=result,
            )
        except Exception as exc:
            return SandboxPageModel(
                world_id=world_id,
                policy_id=policy_id,
                horizon=horizon,
                params=params_final,
                arm_overrides=arm_overrides_final,
                available_worlds=worlds,
                available_policies=available_policies,
                scenario=scenario,
                run_error=str(exc),
            )

    return SandboxPageModel(
        world_id=world_id,
        policy_id=policy_id,
        horizon=horizon,
        params=params_final,
        arm_overrides=arm_overrides_final,
        available_worlds=worlds,
        available_policies=available_policies,
        scenario=scenario,
    )
