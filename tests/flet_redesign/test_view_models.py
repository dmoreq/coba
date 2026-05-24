"""Tests for route UI view-model composition."""

from __future__ import annotations

from web.ui.preferences import UserPreferences
from web.ui.view_models import build_route_ui_model


def test_home_route_model_has_no_three_pane_layout() -> None:
    model = build_route_ui_model("/", prefs=UserPreferences())
    assert model.layout is None
    assert model.scene_panel is None
    assert model.treatment_cards == ()


def test_lesson_route_model_includes_three_panes_and_cards() -> None:
    model = build_route_ui_model(
        "/lesson",
        prefs=UserPreferences(world_id="rural_clinic", policy_id="linucb", speed="1x"),
    )
    assert model.layout is not None
    assert model.scene_panel is not None
    assert len(model.treatment_cards) == 3
    assert len(model.param_controls) >= 1
    assert model.lesson_panel is not None
    assert model.lesson_panel.stage_index == 1
    assert "linucb_debug" in model.capability_debug_views


def test_route_model_uses_selected_world_from_preferences() -> None:
    model = build_route_ui_model(
        "/arena",
        prefs=UserPreferences(world_id="newsfeed", policy_id="epsilon_greedy", speed="1x"),
    )
    assert model.scene_panel is not None
    assert model.scene_panel.world_title == "NewsFeed"
    assert model.arena_metrics is not None
