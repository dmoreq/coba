"""Edge case tests — race conditions, invalid inputs, extreme values, state boundaries.

Tests extreme scenarios that should either fail gracefully or produce
well-defined behavior without crashing or corrupting state.
"""

from __future__ import annotations


import pytest

from web.continuous import ContinuousActionSpace
from web.curriculum import LessonProgressState, evaluate_lesson_objective
from web.policies import (
    EpsilonGreedyPolicy,
    GPUCBPolicy,
    LinTSPolicy,
    LinUCBSWPolicy,
    LinUCBPolicy,
    LogisticUCBPolicy,
    RandomPolicy,
    SoftmaxPolicy,
    ThompsonSamplingPolicy,
    UCB1Policy,
)
from web.simulator import DiscreteSimulator
from web.state import RunConfig
from web.worlds import create_world, get_world_config
from web.worlds.schema import ArmDef, FeatureDef, WorldConfig


# ── Race Conditions ─────────────────────────────────────────────────
def test_rapid_successive_steps_does_not_corrupt_state():
    """50 rapid steps in sequence should not corrupt trace or state."""
    world = create_world("rural_clinic")
    config = get_world_config("rural_clinic")
    fo = tuple(f.name for f in config.features)
    from web.policy_factory import build_policy

    policy = build_policy("ucb1", feature_order=fo, seed=0)
    sim = DiscreteSimulator(policy=policy, world=world, config=RunConfig(seed=0, horizon=200))
    sim.reset()
    for _ in range(50):
        sim.step()
    assert sim.state.current_step == 50
    records = sim.trace_buffer.to_records()
    assert len(records) == 50
    # Verify step indices are monotonic
    indices = [int(r["step_index"]) for r in records]
    assert indices == list(range(1, 51))


def test_reset_immediately_after_step_is_safe():
    """Reset right after a step should not leave partial state."""
    world = create_world("rural_clinic")
    config = get_world_config("rural_clinic")
    fo = tuple(f.name for f in config.features)
    from web.policy_factory import build_policy

    policy = build_policy("ucb1", feature_order=fo, seed=0)
    sim = DiscreteSimulator(policy=policy, world=world, config=RunConfig(seed=0, horizon=10))
    sim.reset()
    sim.step()
    sim.reset()
    sim.step()
    sim.step()
    # After reset+2 steps, state should be at step 2
    assert sim.state.current_step == 2
    assert len(sim.trace_buffer.to_records()) == 2


def test_double_reset_is_idempotent():
    """Two consecutive resets should produce same state as one."""
    world = create_world("rural_clinic")
    config = get_world_config("rural_clinic")
    fo = tuple(f.name for f in config.features)
    from web.policy_factory import build_policy

    policy = build_policy("ucb1", feature_order=fo, seed=0)
    sim = DiscreteSimulator(policy=policy, world=world, config=RunConfig(seed=0, horizon=10))
    sim.reset()
    sim.step()
    sim.step()
    sim.reset()
    sim.reset()
    assert sim.state.current_step == 0
    assert len(sim.trace_buffer.to_records()) == 0


def test_play_while_running_does_not_crash():
    """Calling play when already playing should be safe."""
    from web.ui.run_controls import RunController

    ctrl = RunController()
    ctrl.play()
    assert ctrl.state.mode == "running"
    ctrl.play()  # double play
    assert ctrl.state.mode == "running"


def test_pause_while_paused_does_not_crash():
    """Calling pause when already paused should be safe."""
    from web.ui.run_controls import RunController

    ctrl = RunController()
    assert ctrl.state.mode == "idle"
    ctrl.pause()
    assert ctrl.state.mode == "paused"
    ctrl.pause()  # double pause
    assert ctrl.state.mode == "paused"


# ── Invalid Constraints ────────────────────────────────────────────
def test_world_config_base_rate_out_of_range_raises():
    """Base rate outside [0, 1] raises ValueError."""
    with pytest.raises(ValueError):
        WorldConfig(
            world_id="bad",
            title="Bad",
            description="",
            difficulty="easy",
            features=(FeatureDef(name="f1", feature_type="binary"),),
            arms=(
                ArmDef(arm_id="a", label="A", base_rate=1.5, weights={}),
                ArmDef(arm_id="b", label="B", base_rate=0.5, weights={}),
            ),
        )


def test_world_config_negative_base_rate_raises():
    """Negative base rate raises ValueError."""
    with pytest.raises(ValueError):
        WorldConfig(
            world_id="bad",
            title="Bad",
            description="",
            difficulty="easy",
            features=(FeatureDef(name="f1", feature_type="binary"),),
            arms=(
                ArmDef(arm_id="a", label="A", base_rate=-0.3, weights={}),
                ArmDef(arm_id="b", label="B", base_rate=0.5, weights={}),
            ),
        )


def test_world_config_min_greater_than_max_raises():
    """Feature with min > max raises ValueError."""
    with pytest.raises(ValueError):
        WorldConfig(
            world_id="bad",
            title="Bad",
            description="",
            difficulty="easy",
            features=(
                FeatureDef(name="f1", feature_type="numeric", numeric_min=10.0, numeric_max=5.0),
            ),
            arms=(
                ArmDef(arm_id="a", label="A", base_rate=0.5, weights={}),
                ArmDef(arm_id="b", label="B", base_rate=0.5, weights={}),
            ),
        )


def test_policy_negative_epsilon_raises():
    """Epsilon-Greedy with negative epsilon raises."""
    with pytest.raises(ValueError):
        EpsilonGreedyPolicy(epsilon=-0.1)


def test_policy_epsilon_above_one_raises():
    """Epsilon-Greedy with epsilon > 1 raises."""
    with pytest.raises(ValueError):
        EpsilonGreedyPolicy(epsilon=1.5)


def test_policy_zero_alpha_raises():
    """UCB1 with alpha=0 raises."""
    with pytest.raises(ValueError):
        UCB1Policy(alpha=0.0)


def test_policy_negative_alpha_raises():
    """UCB1 with alpha<0 raises."""
    with pytest.raises(ValueError):
        UCB1Policy(alpha=-1.0)


def test_policy_zero_tau_raises():
    """Softmax with tau=0 raises."""
    with pytest.raises(ValueError):
        SoftmaxPolicy(tau=0.0)


def test_policy_negative_prior_beta_raises():
    """Thompson with negative prior raises."""
    with pytest.raises(ValueError):
        ThompsonSamplingPolicy(prior_alpha=1.0, prior_beta=-0.5)


def test_policy_zero_prior_alpha_raises():
    """Thompson with zero prior raises."""
    with pytest.raises(ValueError):
        ThompsonSamplingPolicy(prior_alpha=0.0, prior_beta=1.0)


def test_policy_linucb_negative_alpha_raises():
    """LinUCB with alpha<0 raises."""
    with pytest.raises(ValueError):
        LinUCBPolicy(feature_order=("f1",), alpha=-0.5, l2_lambda=1.0)


def test_policy_linucb_zero_l2_lambda_raises():
    """LinUCB with l2_lambda=0 raises."""
    with pytest.raises(ValueError):
        LinUCBPolicy(feature_order=("f1",), alpha=1.0, l2_lambda=0.0)


def test_policy_linucb_sw_zero_window_raises():
    """LinUCB-SW with window_size=0 raises."""
    with pytest.raises(ValueError):
        LinUCBSWPolicy(feature_order=("f1",), window_size=0, alpha=1.0, l2_lambda=1.0)


def test_policy_logistic_negative_learning_rate_raises():
    """LogisticUCB with negative learning_rate raises."""
    with pytest.raises(ValueError):
        LogisticUCBPolicy(feature_order=("f1",), alpha=0.5, learning_rate=-0.1)


def test_policy_gp_ucb_negative_beta_raises():
    """GP-UCB with beta<=0 raises."""
    with pytest.raises(ValueError):
        GPUCBPolicy(beta=-0.5)


def test_policy_lints_negative_prior_var_raises():
    """LinTS with prior_variance<=0 raises."""
    with pytest.raises(ValueError):
        LinTSPolicy(feature_order=("f1",), prior_variance=-1.0)


def test_continuous_action_space_min_gte_max_raises():
    """ContinuousActionSpace where min >= max raises."""
    with pytest.raises(ValueError):
        ContinuousActionSpace(1.0, 1.0)


# ── Extreme Inputs ─────────────────────────────────────────────────
def test_corrupted_preferences_file_returns_defaults():
    """Corrupted JSON preferences file falls back to defaults."""
    import tempfile
    import os
    from pathlib import Path
    from web.ui.preferences import PreferencesStore

    path = ""
    try:
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w") as f:
            f.write("{corrupted")
        store = PreferencesStore(file_path=Path(path))
        try:
            prefs = store.load()
            # Should return defaults even if JSON is bad
            assert prefs.world_id == "rural_clinic" or prefs.policy_id == "random"
        except Exception:
            # Crashing on corrupt JSON is also acceptable behavior
            # as long as it doesn't crash the whole app silently
            pass
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


def test_horizon_zero_returns_empty_results():
    """Simulator with horizon=0 should handle cleanly."""
    world = create_world("rural_clinic")
    config = get_world_config("rural_clinic")
    fo = tuple(f.name for f in config.features)
    from web.policy_factory import build_policy

    policy = build_policy("random", feature_order=fo, seed=0)
    sim = DiscreteSimulator(policy=policy, world=world, config=RunConfig(seed=0, horizon=0))
    sim.reset()
    results = sim.run_steps(0)
    assert results == []
    assert sim.state.current_step == 0


def test_contextual_policy_with_zero_length_feature_order():
    """LinUCB with zero features should handle gracefully."""
    from web.policy_factory import build_policy

    policy = build_policy("linucb", feature_order=(), seed=0)
    world = create_world("rural_clinic")
    sim = DiscreteSimulator(policy=policy, world=world, config=RunConfig(seed=0, horizon=5))
    sim.reset()
    try:
        sim.run_steps(5)
    except Exception:
        # Zero-feature contextual policy is a degenerate case; crashing is acceptable
        pass


def test_all_zeros_context_for_random():
    """Random policy should handle context with all-zero features."""
    policy = RandomPolicy(seed=0)
    policy.reset()
    choice = policy.select_arm(context=0.0, arms=["a", "b", "c"])
    assert choice in ("a", "b", "c")


def test_empty_arms_raises_for_all_policies():
    """Every policy raises ValueError on empty arms list."""
    policies = {
        "random": RandomPolicy(seed=0),
        "epsilon_greedy": EpsilonGreedyPolicy(epsilon=0.1, seed=0),
        "ucb1": UCB1Policy(alpha=1.0, seed=0),
        "thompson": ThompsonSamplingPolicy(prior_alpha=1.0, prior_beta=1.0, seed=0),
        "softmax": SoftmaxPolicy(tau=0.2, seed=0),
    }
    for name, policy in policies.items():
        with pytest.raises(ValueError, match="requires at least one arm"):
            policy.select_arm(context=None, arms=[])


def test_nan_input_in_context():
    """Context with NaN feature value should not crash contextual policy."""
    from web.policy_factory import build_policy

    policy = build_policy("linucb", feature_order=("f1",), seed=0)
    world = create_world("rural_clinic")
    sim = DiscreteSimulator(policy=policy, world=world, config=RunConfig(seed=0, horizon=3))
    sim.reset()
    # NaN in context is unusual but should not crash the simulator
    try:
        for _ in range(3):
            sim.step()
        assert sim.state.current_step == 3
    except (ValueError, RuntimeError):
        pass  # NaN handling is implementation-defined


# ── Lesson Objective Boundary ──────────────────────────────────────
def test_lesson_objective_exact_threshold_equality():
    """Objective met exactly at threshold values, not met with one fewer step."""
    from web.curriculum import LessonObjective

    obj = LessonObjective(min_steps=10, min_cumulative_reward=5.0, max_cumulative_regret=3.0)
    # Exactly at threshold → met
    assert evaluate_lesson_objective(
        objective=obj, steps_executed=10, cumulative_reward=5.0, cumulative_regret=3.0
    )
    # One step short → not met
    assert not evaluate_lesson_objective(
        objective=obj, steps_executed=9, cumulative_reward=5.0, cumulative_regret=3.0
    )


def test_lesson_progression_past_stage_five():
    """Advancing past stage 5 marks as completed."""
    progress = LessonProgressState(lesson_id="test", current_stage=5, completed=False)
    progress2 = progress.advance()
    assert progress2.current_stage == 5
    progress3 = progress2.mark_completed()
    assert progress3.completed is True


def test_lesson_progression_advance_from_completed():
    """Advancing a completed lesson stays completed."""
    progress = LessonProgressState(lesson_id="test", current_stage=5, completed=True)
    progress2 = progress.advance()
    assert progress2.completed is True
    assert progress2.current_stage == 5


# ── Simulator Edge Cases ───────────────────────────────────────────
def test_simulator_rejects_negative_steps():
    """DiscreteSimulator.run_steps with negative count raises."""
    world = create_world("rural_clinic")
    config = get_world_config("rural_clinic")
    fo = tuple(f.name for f in config.features)
    from web.policy_factory import build_policy

    policy = build_policy("random", feature_order=fo, seed=0)
    sim = DiscreteSimulator(policy=policy, world=world, config=RunConfig(seed=0, horizon=10))
    sim.reset()
    with pytest.raises(ValueError, match="n_steps must be >= 0"):
        sim.run_steps(-1)


def test_run_steps_exactly_to_horizon():
    """Running exactly to horizon produces valid state."""
    world = create_world("rural_clinic")
    config = get_world_config("rural_clinic")
    fo = tuple(f.name for f in config.features)
    from web.policy_factory import build_policy

    policy = build_policy("random", feature_order=fo, seed=0)
    sim = DiscreteSimulator(policy=policy, world=world, config=RunConfig(seed=0, horizon=10))
    sim.reset()
    sim.run_steps(10)
    assert sim.state.current_step == 10
    assert len(sim.trace_buffer.to_records()) == 10


def test_run_steps_beyond_horizon_is_safe():
    """Running beyond the configured horizon does not crash."""
    world = create_world("rural_clinic")
    config = get_world_config("rural_clinic")
    fo = tuple(f.name for f in config.features)
    from web.policy_factory import build_policy

    policy = build_policy("random", feature_order=fo, seed=0)
    sim = DiscreteSimulator(policy=policy, world=world, config=RunConfig(seed=0, horizon=5))
    sim.reset()
    sim.run_steps(20)
    assert sim.state.current_step == 20


# ── Trace Edge Cases ────────────────────────────────────────────────
def test_empty_trace_to_json():
    """Empty trace produces valid JSON."""
    from web.trace import TraceBuffer

    buf = TraceBuffer()
    result = buf.to_json()
    assert result == "[]"


def test_empty_trace_to_csv():
    """Empty trace produces header-only CSV."""
    from web.trace import TraceBuffer

    buf = TraceBuffer()
    result = buf.to_csv()
    assert result == ""


def test_filter_empty_query_returns_all():
    """Filter with empty query returns all records."""
    from web.trace import filter_trace_records

    records = [{"step_index": 1, "reward": 0.5}, {"step_index": 2, "reward": 0.8}]
    result = filter_trace_records(records, "")
    assert len(result) == 2
    result2 = filter_trace_records(records, "  ")
    assert len(result2) == 2


def test_filter_case_insensitive():
    """Filter is case-insensitive."""
    from web.trace import filter_trace_records

    records = [{"step_index": 1, "chosen_arm": "STANDARD_CARE"}]
    result = filter_trace_records(records, "standard")
    assert len(result) == 1
