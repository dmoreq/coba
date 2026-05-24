"""Tests for Flet redesign state containers."""

from __future__ import annotations

from coba.flet_redesign.contracts import SimulationStepResult
from coba.flet_redesign.state import ArmState, RunConfig, SimulationState


def test_arm_state_mean_reward_handles_zero_pulls() -> None:
    arm_state = ArmState(arm="a")
    assert arm_state.mean_reward == 0.0


def test_arm_state_mean_reward_computes_average() -> None:
    arm_state = ArmState(arm="a", pulls=4, reward_sum=2.0)
    assert arm_state.mean_reward == 0.5


def test_simulation_state_append_step_updates_aggregates() -> None:
    state = SimulationState(config=RunConfig(seed=7, horizon=20))
    step = SimulationStepResult(
        step_index=1,
        context={"x": 1.0},
        chosen_arm="arm_a",
        reward=0.8,
        cumulative_reward=0.8,
        cumulative_regret=0.1,
    )
    state.append_step(step)
    assert state.current_step == 1
    assert state.cumulative_reward == 0.8
    assert state.cumulative_regret == 0.1
    assert len(state.trace) == 1


def test_simulation_state_reset_preserves_config() -> None:
    config = RunConfig(seed=99, horizon=300)
    state = SimulationState(config=config, current_step=20, cumulative_reward=7.1, cumulative_regret=2.4)
    state.trace.append(
        SimulationStepResult(
            step_index=20,
            context={},
            chosen_arm="arm_b",
            reward=0.2,
            cumulative_reward=7.1,
            cumulative_regret=2.4,
        )
    )
    state.reset()
    assert state.config is config
    assert state.current_step == 0
    assert state.cumulative_reward == 0.0
    assert state.cumulative_regret == 0.0
    assert not state.is_running
    assert state.trace == []
