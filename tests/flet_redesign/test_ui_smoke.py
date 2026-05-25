"""UI smoke tests: route model builders, shell stack, navigation, param controls."""

from __future__ import annotations

from web.router import AppRoute, normalize_route

from web.ui.preferences import UserPreferences
from web.ui.view_models import build_route_ui_model


def test_home_route_model() -> None:
    model = build_route_ui_model("/", prefs=UserPreferences())
    assert model.layout is None
    assert model.heading
    assert model.title
    assert model.scene_panel is None


def test_lesson_route_model_all_policies() -> None:
    for policy_id in [
        "random",
        "epsilon_greedy",
        "ucb1",
        "thompson",
        "softmax",
        "linucb",
        "logistic_ucb",
        "gp_ucb",
        "bootstrapped_ensemble",
        "linucb_hybrid",
        "tree_ucb",
        "tree_ts",
    ]:
        prefs = UserPreferences(world_id="rural_clinic", policy_id=policy_id, speed="1x")
        model = build_route_ui_model("/lesson", prefs=prefs)
        assert model.layout is not None
        assert model.lesson_panel is not None
        assert model.lesson_panel.stage_index in (1, 2, 3, 4, 5)


def test_arena_route_model() -> None:
    model = build_route_ui_model(
        "/arena", prefs=UserPreferences(world_id="moviematch", policy_id="ucb1", speed="1x")
    )
    assert model.layout is not None
    assert model.scene_panel is not None
    assert model.arena_metrics is not None
    assert len(model.treatment_cards) == 3


def test_sandbox_route_model() -> None:
    model = build_route_ui_model(
        "/sandbox", prefs=UserPreferences(world_id="newsfeed", policy_id="thompson", speed="1x")
    )
    assert model.layout is not None
    assert model.scene_panel is not None
    assert model.scene_panel.world_title == "NewsFeed"


def test_normalize_route_edge_cases() -> None:
    assert normalize_route(None) == AppRoute.HOME
    assert normalize_route("") == AppRoute.HOME
    assert normalize_route("/unknown") == AppRoute.HOME
    assert normalize_route("lesson") == AppRoute.LESSON
    assert normalize_route("/lesson/extra") == AppRoute.LESSON
    assert normalize_route("/arena") == AppRoute.ARENA
    assert normalize_route("sandbox/custom") == AppRoute.SANDBOX


def test_preferences_defaults() -> None:
    prefs = UserPreferences()
    assert prefs.world_id == "rural_clinic"
    assert prefs.policy_id == "random"
    assert prefs.speed == "1x"


def test_param_controls_for_all_supported_policies() -> None:
    from web.ui.param_controls import default_policy_param_controls

    for pid in ["epsilon_greedy", "ucb1", "softmax", "linucb", "linucb_sw", "logistic_ucb", "cats"]:
        controls = default_policy_param_controls(pid)
        assert isinstance(controls, tuple)
        assert len(controls) >= 1, f"No controls for {pid}"


def test_view_model_with_live_state() -> None:
    prefs = UserPreferences(world_id="rural_clinic", policy_id="ucb1", speed="1x")
    model = build_route_ui_model(
        "/arena",
        prefs=prefs,
        trace_records=(),
        sim_step_index=50,
        sim_cumulative_reward=25.0,
        sim_cumulative_regret=10.0,
    )
    assert model.arena_metrics is not None
    assert len(model.treatment_cards) == 3


def test_context_inspection_present_for_contextual_policy() -> None:
    prefs = UserPreferences(world_id="rural_clinic", policy_id="linucb", speed="1x")
    model = build_route_ui_model("/lesson", prefs=prefs)
    assert model.context_inspection is not None
    assert len(model.context_inspection.feature_order) >= 1
    assert len(model.context_inspection.feature_values) >= 1
