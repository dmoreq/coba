"""E2E tests for lesson, sandbox, comparison, and home routes."""

from __future__ import annotations

from web.analysis import run_policy_comparison, summarize_comparison_runs
from web.ui.preferences import UserPreferences
from web.ui.view_models import build_route_ui_model


def test_lesson_route_all_policies_have_lesson() -> None:
    """Every mapped lesson policy produces a non-null lesson panel."""
    for policy_id in [
        "random",
        "epsilon_greedy",
        "ucb1",
        "thompson",
        "softmax",
        "linucb",
        "lints",
        "logistic_ucb",
    ]:
        prefs = UserPreferences(world_id="rural_clinic", policy_id=policy_id, speed="1x")
        model = build_route_ui_model("/lesson", prefs=prefs)
        assert model.lesson_panel is not None
        assert model.lesson_panel.stage_index >= 1


def test_lesson_panel_has_five_stages() -> None:
    """The lesson panel shows stage X out of 5."""
    prefs = UserPreferences(world_id="rural_clinic", policy_id="ucb1", speed="1x")
    model = build_route_ui_model("/lesson", prefs=prefs)
    assert model.lesson_panel is not None
    assert model.lesson_panel.stage_index <= 5


def test_sandbox_route_returns_max_controls() -> None:
    """Sandbox route provides param controls for tuning."""
    prefs = UserPreferences(world_id="rural_clinic", policy_id="epsilon_greedy", speed="1x")
    model = build_route_ui_model("/sandbox", prefs=prefs)
    assert model.param_controls is not None
    assert len(model.param_controls) >= 1


def test_comparison_orchestrator_multi_policy() -> None:
    """Comparison route supports running multiple policies concurrently."""
    results = run_policy_comparison(
        world_id="rural_clinic",
        policy_ids=["random", "ucb1", "thompson"],
        seed=42,
        horizon=50,
    )
    assert len(results) == 3
    rewards = [r.cumulative_reward for r in results]
    assert all(r >= 0 for r in rewards)


def test_comparison_summaries_ordered_by_reward() -> None:
    """Comparison summaries are sorted descending by mean reward."""
    results = run_policy_comparison(
        world_id="rural_clinic",
        policy_ids=["random", "ucb1"],
        seed=7,
        horizon=30,
    )
    summaries = summarize_comparison_runs(results)
    assert len(summaries) == 2
    assert summaries[0].mean_reward >= summaries[1].mean_reward


def test_home_route_no_scene_or_layout() -> None:
    """Home route has no scene panel or layout (it's a landing page)."""
    prefs = UserPreferences()
    model = build_route_ui_model("/", prefs=prefs)
    assert model.layout is None
    assert model.scene_panel is None


def test_home_route_has_title_and_description() -> None:
    """Home route contains heading and description text."""
    prefs = UserPreferences()
    model = build_route_ui_model("/", prefs=prefs)
    assert model.heading
    assert model.description


def test_world_change_affects_scene_panel() -> None:
    """Switching worlds produces different scene panel titles."""
    prefs_a = UserPreferences(world_id="rural_clinic", policy_id="random", speed="1x")
    prefs_b = UserPreferences(world_id="moviematch", policy_id="random", speed="1x")

    model_a = build_route_ui_model("/arena", prefs=prefs_a)
    model_b = build_route_ui_model("/arena", prefs=prefs_b)

    assert model_a.scene_panel is not None
    assert model_b.scene_panel is not None
    assert model_a.scene_panel.world_title != model_b.scene_panel.world_title


def test_policy_change_affects_param_controls() -> None:
    """Different policies expose different param controls."""
    prefs_ucb = UserPreferences(world_id="rural_clinic", policy_id="ucb1", speed="1x")
    prefs_eps = UserPreferences(world_id="rural_clinic", policy_id="epsilon_greedy", speed="1x")

    model_ucb = build_route_ui_model("/arena", prefs=prefs_ucb)
    model_eps = build_route_ui_model("/arena", prefs=prefs_eps)

    ucb_keys = {s.key for s in (model_ucb.param_controls or ())}
    eps_keys = {s.key for s in (model_eps.param_controls or ())}
    assert ucb_keys != eps_keys
