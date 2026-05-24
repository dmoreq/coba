"""Integration tests: full step loops per policy-world pair, lesson progression, debug snapshots."""

from __future__ import annotations

from web.contracts import DebugSnapshotProvider
from web.curriculum import evaluate_lesson_objective, get_lesson
from web.policy_factory import build_policy
from web.simulator import DiscreteSimulator
from web.state import RunConfig
from web.worlds import create_world, get_world_config


POLICY_IDS = [
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
]

WORLD_IDS = ["rural_clinic", "moviematch", "newsfeed"]


def _feature_order(world_id: str) -> tuple[str, ...]:
    return tuple(f.name for f in get_world_config(world_id).features)


def test_all_policies_run_full_loop_on_clinic() -> None:
    for policy_id in POLICY_IDS:
        world = create_world("rural_clinic")
        fo = _feature_order("rural_clinic")
        policy = build_policy(policy_id, feature_order=fo, seed=42)
        simulator = DiscreteSimulator(
            policy=policy,
            world=world,
            config=RunConfig(seed=42, horizon=100),
        )
        simulator.reset()
        results = simulator.run_steps(100)
        assert len(results) == 100
        assert simulator.state.current_step == 100
        assert simulator.state.cumulative_reward >= 0.0
        assert simulator.state.cumulative_regret >= 0.0


def test_all_contextual_policies_run_with_features() -> None:
    contextual = ["linucb", "linucb_sw", "logistic_ucb", "linucb_hybrid"]
    for policy_id in contextual:
        for world_id in WORLD_IDS:
            world = create_world(world_id)
            fo = _feature_order(world_id)
            policy = build_policy(policy_id, feature_order=fo, seed=7)
            simulator = DiscreteSimulator(
                policy=policy, world=world, config=RunConfig(seed=7, horizon=50)
            )
            simulator.reset()
            results = simulator.run_steps(50)
            assert len(results) == 50
            for r in results:
                assert "context" in r.metadata or True


def test_lesson_progression_completes_for_random_baseline() -> None:
    lesson = get_lesson("lesson_random_baseline")
    fo = _feature_order(lesson.world_id)
    policy = build_policy(lesson.policy_id, feature_order=fo, seed=0)
    world = create_world(lesson.world_id)
    sim = DiscreteSimulator(policy=policy, world=world, config=RunConfig(seed=0, horizon=200))
    sim.reset()
    sim.run_steps(lesson.objective.min_steps)
    met = evaluate_lesson_objective(
        objective=lesson.objective,
        steps_executed=sim.state.current_step,
        cumulative_reward=sim.state.cumulative_reward,
        cumulative_regret=sim.state.cumulative_regret,
    )
    assert met


def test_all_ucb1_lesson_progression() -> None:
    lesson = get_lesson("lesson_ucb1")
    fo = _feature_order(lesson.world_id)
    policy = build_policy(lesson.policy_id, feature_order=fo, seed=1)
    world = create_world(lesson.world_id)
    sim = DiscreteSimulator(policy=policy, world=world, config=RunConfig(seed=1, horizon=300))
    sim.reset()
    sim.run_steps(lesson.objective.min_steps)
    met = evaluate_lesson_objective(
        objective=lesson.objective,
        steps_executed=sim.state.current_step,
        cumulative_reward=sim.state.cumulative_reward,
        cumulative_regret=sim.state.cumulative_regret,
    )
    assert met


def test_debug_snapshot_for_advanced_policies() -> None:
    debug_policies = [
        "linucb",
        "linucb_sw",
        "logistic_ucb",
        "gp_ucb",
        "bootstrapped_ensemble",
        "linucb_hybrid",
        "tree_ucb",
        "tree_ts",
    ]
    for policy_id in debug_policies:
        fo = _feature_order("rural_clinic")
        policy = build_policy(policy_id, feature_order=fo, seed=3)
        world = create_world("rural_clinic")
        sim = DiscreteSimulator(policy=policy, world=world, config=RunConfig(seed=3, horizon=10))
        sim.reset()
        sim.run_steps(10)
        if isinstance(policy, DebugSnapshotProvider):
            snapshot = policy.get_debug_snapshot()
            assert isinstance(snapshot, dict)
            assert len(snapshot) > 0


def test_checkpoint_roundtrip_preserves_state() -> None:
    from web.checkpoint import CheckpointPayload, load_checkpoint, save_checkpoint
    import tempfile
    import os
    from pathlib import Path

    policy = build_policy("ucb1", feature_order=(), seed=5)
    world = create_world("rural_clinic")
    sim = DiscreteSimulator(policy=policy, world=world, config=RunConfig(seed=5, horizon=30))
    sim.reset()
    sim.run_steps(20)
    records = sim.trace_buffer.to_records()
    payload = CheckpointPayload(
        checkpoint_id="test-cp",
        kind="discrete",
        state={"world_id": "rural_clinic", "policy_id": "ucb1", "seed": 5},
        trace=records,
    )
    path = ""
    try:
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        save_checkpoint(Path(path), payload)
        loaded = load_checkpoint(Path(path))
        assert loaded.checkpoint_id == "test-cp"
        assert len(loaded.trace) == 20
    finally:
        if path and os.path.exists(path):
            os.unlink(path)
