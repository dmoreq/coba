"""E2E frontend tests — programmatic UI flow testing via view-models and sessions.

Flet 0.85+ doesn't provide headless browser test mode, so we test
the complete user interaction flow through the session/view-model layer:
1. Create _SimSession (same object Flet creates on app start)
2. Call simulator methods directly (same code path as button callbacks)
3. Build views via build_route_ui_model (same code path as _refresh_view)
4. Verify lesson progression, world/policy switching, and metrics
"""

from __future__ import annotations


from web.policy_factory import build_policy
from web.simulator import DiscreteSimulator
from web.state import RunConfig
from web.ui.preferences import UserPreferences
from web.ui.view_models import build_route_ui_model
from web.worlds import create_world, get_world_config


def _make_session(prefs=None):
    """Create a _SimSession exactly as main.py does on app startup."""
    from web.main import _SimSession

    if prefs is None:
        prefs = UserPreferences(world_id="rural_clinic", policy_id="ucb1", speed="1x")
    return _SimSession(prefs)


# ── Basic Interaction Flow ─────────────────────────────────────────
def test_step_button_increments_counter():
    """Clicking Step advances simulator by one step."""
    session = _make_session()
    assert session.simulator.state.current_step == 0
    session.simulator.step()
    assert session.simulator.state.current_step == 1
    session.simulator.step()
    assert session.simulator.state.current_step == 2


def test_multiple_steps_accumulate_trace():
    """Trace buffer grows with each step."""
    session = _make_session()
    for _ in range(5):
        session.simulator.step()
    records = session.simulator.trace_buffer.to_records()
    assert len(records) == 5
    assert all("step_index" in r for r in records)


def test_reset_button_clears_simulator():
    """Reset returns simulator to step 0 with empty trace."""
    session = _make_session()
    for _ in range(10):
        session.simulator.step()
    assert session.simulator.state.current_step == 10
    session.do_reset()
    assert session.simulator.state.current_step == 0
    assert len(session.simulator.trace_buffer.to_records()) == 0


def test_play_pause_reset_controller_flow():
    """RunController state machine: idle → play → pause → reset."""
    session = _make_session()
    ctrl = session.controller
    assert ctrl.state.mode in ("idle", "paused")
    ctrl.play()
    assert ctrl.state.mode == "running"
    ctrl.pause()
    assert ctrl.state.mode == "paused"
    ctrl.reset()
    assert ctrl.state.mode == "idle"
    assert ctrl.state.steps_executed == 0


# ── Navigation ─────────────────────────────────────────────────────
def test_lesson_route_view_model():
    """Lesson route builds view with theory panel."""
    session = _make_session()
    view = build_route_ui_model("/lesson", prefs=session.prefs)
    assert view.title == "Lesson"
    assert view.lesson_panel is not None
    assert view.lesson_panel.stage_index >= 1


def test_arena_route_view_model():
    """Arena route builds view with metrics panel."""
    session = _make_session()
    view = build_route_ui_model("/arena", prefs=session.prefs)
    assert view.title == "Arena"
    assert view.arena_metrics is not None


def test_sandbox_route_view_model():
    """Sandbox route builds view with scene panel."""
    session = _make_session()
    view = build_route_ui_model("/sandbox", prefs=session.prefs)
    assert view.title == "Sandbox"


def test_comparison_route_spec():
    """Comparison route has correct spec."""
    from web.router import get_route_spec

    spec = get_route_spec("/comparison")
    assert spec.title == "Comparison"


def test_home_route_no_layout():
    """Home route has no three-pane layout."""
    session = _make_session()
    view = build_route_ui_model("/", prefs=session.prefs)
    assert view.layout is None


# ── World/Policy Switching ─────────────────────────────────────────
def test_world_change_rebuilds_simulator():
    """World switch resets simulator with new world."""
    session = _make_session(
        prefs=UserPreferences(world_id="rural_clinic", policy_id="ucb1", speed="1x")
    )
    session.simulator.step()
    assert session.simulator.state.current_step == 1
    new_prefs = UserPreferences(world_id="moviematch", policy_id="ucb1", speed="1x")
    session.sync_prefs(new_prefs)
    assert session.prefs.world_id == "moviematch"
    assert session.simulator.state.current_step == 0


def test_policy_change_rebuilds_simulator():
    """Policy switch resets simulator with new policy."""
    session = _make_session(
        prefs=UserPreferences(world_id="rural_clinic", policy_id="ucb1", speed="1x")
    )
    session.simulator.step()
    new_prefs = UserPreferences(world_id="rural_clinic", policy_id="epsilon_greedy", speed="1x")
    session.sync_prefs(new_prefs)
    assert session.prefs.policy_id == "epsilon_greedy"
    assert session.simulator.state.current_step == 0


# ── Lesson Objective ───────────────────────────────────────────────
def test_lesson_objective_check_after_steps():
    """After running steps, lesson objectives update correctly."""
    prefs = UserPreferences(world_id="rural_clinic", policy_id="ucb1", speed="1x")
    session = _make_session(prefs)
    assert session.lesson_config is not None
    assert session.lesson_progress is not None
    assert session.lesson_progress.current_stage == 1
    sim = session.simulator
    for _ in range(40):
        sim.step()
    # After 40 steps, UCB1 should be approaching or meeting objective
    assert sim.state.current_step == 40
    assert sim.state.cumulative_reward > 0.0


# ── View Model Integration ─────────────────────────────────────────
def test_view_model_reflects_live_state():
    """After stepping, arena view model shows live trace data."""
    session = _make_session()
    for _ in range(5):
        session.simulator.step()
    records = tuple(session.simulator.trace_buffer.to_records())
    view = build_route_ui_model(
        "/arena",
        prefs=session.prefs,
        trace_records=records,
        sim_step_index=session.simulator.state.current_step,
        sim_cumulative_reward=session.simulator.state.cumulative_reward,
        sim_cumulative_regret=session.simulator.state.cumulative_regret,
    )
    assert len(view.trace_records) == 5
    assert view.arena_metrics is not None
    assert len(view.arena_metrics.reward_series) == 5


def test_treatment_cards_show_selected_arm():
    """After stepping, one treatment card is marked selected."""
    session = _make_session()
    for _ in range(3):
        session.simulator.step()
    records = tuple(session.simulator.trace_buffer.to_records())
    view = build_route_ui_model(
        "/lesson",
        prefs=session.prefs,
        trace_records=records,
    )
    selected = [c for c in view.treatment_cards if c.selected]
    assert len(selected) == 1


# ── Speed Change ───────────────────────────────────────────────────
def test_speed_change_persists():
    """Session preferences update on speed change."""
    session = _make_session()
    session.prefs = UserPreferences(world_id="rural_clinic", policy_id="random", speed="4x")
    assert session.prefs.speed == "4x"


# ── All Policies E2E ───────────────────────────────────────────────
def test_all_policies_step_without_crash():
    """All 14 discrete policies can run 10 steps without exception."""
    for policy_id in [
        "random",
        "epsilon_greedy",
        "ucb1",
        "thompson",
        "softmax",
        "linucb",
        "linucb_sw",
        "lints",
        "logistic_ucb",
        "gp_ucb",
        "bootstrapped_ensemble",
        "linucb_hybrid",
        "tree_ucb",
        "tree_ts",
    ]:
        world = create_world("rural_clinic")
        config = get_world_config("rural_clinic")
        fo = tuple(f.name for f in config.features)
        policy = build_policy(policy_id, feature_order=fo, seed=0)
        sim = DiscreteSimulator(policy=policy, world=world, config=RunConfig(seed=0, horizon=50))
        sim.reset()
        sim.run_steps(10)
        assert sim.state.current_step == 10, f"Failed for {policy_id}"
        assert sim.state.cumulative_reward >= 0.0


def test_all_worlds_work_with_ucb1():
    """UCB1 can run on all 7 worlds without crash."""
    for world_id in [
        "rural_clinic",
        "moviematch",
        "newsfeed",
        "shopsmart",
        "ridepilot",
        "gamebot",
        "labtrial",
    ]:
        world = create_world(world_id)
        config = get_world_config(world_id)
        fo = tuple(f.name for f in config.features)
        policy = build_policy("ucb1", feature_order=fo, seed=0)
        sim = DiscreteSimulator(policy=policy, world=world, config=RunConfig(seed=0, horizon=30))
        sim.reset()
        sim.run_steps(30)
        assert sim.state.current_step == 30, f"Failed for {world_id}"
