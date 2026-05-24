"""Tests for Phase-3 UI layout and controls models."""

from __future__ import annotations

from web.ui.layout import build_three_pane_layout
from web.ui.param_controls import ParamControlSpec, default_policy_param_controls
from web.ui.run_controls import RunController


def test_three_pane_layout_ratios_sum_to_one() -> None:
    layout = build_three_pane_layout()
    assert layout.left.width_ratio > 0
    assert layout.center.width_ratio > 0
    assert layout.right.width_ratio > 0
    assert layout.total_ratio == 1.0


def test_run_controller_step_play_pause_reset_flow() -> None:
    controller = RunController()
    assert controller.state.mode == "idle"

    controller.play()
    assert controller.state.mode == "running"

    controller.step()
    assert controller.state.steps_executed == 1
    assert controller.state.mode == "paused"

    controller.pause()
    assert controller.state.mode == "paused"

    controller.reset()
    assert controller.state.mode == "idle"
    assert controller.state.steps_executed == 0


def test_param_control_spec_validation_for_slider() -> None:
    try:
        ParamControlSpec(
            key="alpha",
            label="Alpha",
            control_type="slider",
            default_value=1.0,
            min_value=0.1,
            max_value=2.0,
            step=0.1,
        )
    except ValueError as exc:  # pragma: no cover
        raise AssertionError(f"valid slider spec unexpectedly failed: {exc}") from exc


def test_default_policy_controls_include_tooltip_payload() -> None:
    controls = default_policy_param_controls("ucb1")
    assert len(controls) == 1
    assert controls[0].tooltip is not None
    assert "sqrt" in controls[0].tooltip.formula


def test_contextual_policy_controls_available() -> None:
    linucb_controls = default_policy_param_controls("linucb")
    assert {control.key for control in linucb_controls} == {"alpha", "l2_lambda"}


def test_continuous_policy_controls_available() -> None:
    cats_controls = default_policy_param_controls("cats")
    assert {control.key for control in cats_controls} == {
        "action_min",
        "action_max",
        "exploration",
    }
