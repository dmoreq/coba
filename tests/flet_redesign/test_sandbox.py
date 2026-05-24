"""Tests for sandbox editor validation and world overrides."""

from __future__ import annotations

import pytest

from web.sandbox import SandboxEditor


def test_sandbox_scenario_defaults() -> None:
    editor = SandboxEditor(world_id="rural_clinic", policy_id="ucb1", horizon=200)
    s = editor.scenario
    assert s.world_id == "rural_clinic"
    assert s.policy_id == "ucb1"
    assert s.horizon == 200
    assert s.params == {}


def test_sandbox_rejects_invalid_horizon() -> None:
    with pytest.raises(ValueError, match="horizon must be > 0"):
        SandboxEditor(world_id="rural_clinic", policy_id="random", horizon=0)
    with pytest.raises(ValueError, match="horizon must be > 0"):
        SandboxEditor(world_id="rural_clinic", policy_id="random", horizon=-1)


def test_sandbox_set_param() -> None:
    editor = SandboxEditor(world_id="rural_clinic", policy_id="ucb1", horizon=100)
    s = editor.set_param("alpha", 2.0)
    assert s.params["alpha"] == 2.0
    s = editor.set_param("l2_lambda", 0.5)
    assert s.params["alpha"] == 2.0
    assert s.params["l2_lambda"] == 0.5


def test_sandbox_build_world_override() -> None:
    editor = SandboxEditor(world_id="rural_clinic", policy_id="ucb1", horizon=100)
    override = editor.build_world_override({"standard_care": 0.65})
    assert override.world_id == "rural_clinic"
    arms = {a.arm_id: a.base_rate for a in override.arms}
    assert arms["standard_care"] == 0.65
    assert arms["targeted_followup"] == 0.52
    assert arms["remote_monitoring"] == 0.48


def test_sandbox_build_world_override_unknown_arm_ignores() -> None:
    editor = SandboxEditor(world_id="moviematch", policy_id="random", horizon=50)
    override = editor.build_world_override({"nonexistent": 0.9, "trending_now": 0.7})
    arms = {a.arm_id: a.base_rate for a in override.arms}
    assert arms["trending_now"] == 0.7
    assert "nonexistent" not in arms


def test_scenario_immutability_through_editor() -> None:
    editor = SandboxEditor(world_id="rural_clinic", policy_id="random", horizon=10)
    s1 = editor.scenario
    editor.set_param("key", "value")
    s2 = editor.scenario
    assert s1 is not s2
    assert s1.params == {}
    assert s2.params == {"key": "value"}
