"""Tests for contextual presets, debug panes, and context inspection view model."""

from __future__ import annotations

from coba.flet_redesign.debug import build_linucb_debug_pane, build_logistic_debug_pane
from coba.flet_redesign.ui.preferences import UserPreferences
from coba.flet_redesign.ui.view_models import build_route_ui_model
from coba.flet_redesign.worlds import list_contextual_presets


def test_contextual_presets_available() -> None:
    presets = list_contextual_presets()
    assert len(presets) >= 3
    assert {preset.world_id for preset in presets} == {"rural_clinic", "moviematch", "newsfeed"}


def test_context_inspection_available_in_route_model() -> None:
    model = build_route_ui_model(
        "/lesson",
        prefs=UserPreferences(world_id="rural_clinic", policy_id="linucb", speed="1x"),
    )
    assert model.context_inspection is not None
    assert len(model.context_inspection.feature_order) == len(
        model.context_inspection.feature_values
    )


def test_contextual_debug_pane_builders() -> None:
    linucb_pane = build_linucb_debug_pane(
        {
            "feature_order": ("x1", "x2"),
            "scores": {"a": 0.8, "b": 0.5},
            "arms": {"a": {}, "b": {}},
        }
    )
    logistic_pane = build_logistic_debug_pane(
        {
            "feature_order": ("x1", "x2"),
            "scores": {"a": 0.7, "b": 0.6},
            "arms": {"a": {}, "b": {}},
        }
    )
    assert linucb_pane.title == "LinUCB Debug"
    assert logistic_pane.title == "Logistic Debug"
