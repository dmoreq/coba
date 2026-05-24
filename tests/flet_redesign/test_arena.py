"""Tests for arena metrics and run snapshot comparison."""

from __future__ import annotations

from web.arena import ArenaRunStore, build_arena_metrics


def _records() -> list[dict[str, object]]:
    return [
        {
            "step_index": 1,
            "chosen_arm": "a",
            "cumulative_reward": 1.0,
            "cumulative_regret": 0.0,
            "metadata": {"uncertainty": 0.2},
        },
        {
            "step_index": 2,
            "chosen_arm": "b",
            "cumulative_reward": 1.0,
            "cumulative_regret": 0.8,
            "metadata": {"uncertainty": 0.3},
        },
    ]


def test_build_arena_metrics_from_trace_records() -> None:
    metrics = build_arena_metrics(_records())
    assert len(metrics.reward_series) == 2
    assert len(metrics.regret_series) == 2
    assert metrics.arm_pull_counts == {"a": 1, "b": 1}
    assert metrics.uncertainty_series[0].value == 0.2


def test_run_store_tracks_previous_and_current_snapshots() -> None:
    store = ArenaRunStore()
    first = store.build_snapshot(
        run_id="run1",
        policy_id="random",
        world_id="rural_clinic",
        cumulative_reward=10.0,
        cumulative_regret=5.0,
        replay_payload={"steps": []},
    )
    second = store.build_snapshot(
        run_id="run2",
        policy_id="ucb1",
        world_id="rural_clinic",
        cumulative_reward=15.0,
        cumulative_regret=3.0,
        replay_payload={"steps": []},
    )
    store.commit(first)
    assert store.current is first
    assert store.previous is None

    store.commit(second)
    assert store.current is second
    assert store.previous is first
