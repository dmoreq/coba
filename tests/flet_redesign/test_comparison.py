"""Tests for comparison orchestrator, stats, and snapshot diff."""

from __future__ import annotations

from web.analysis import run_batch_comparison, run_policy_comparison, summarize_comparison_runs


def test_orchestrator_deterministic_equality() -> None:
    a = run_policy_comparison(
        world_id="rural_clinic",
        policy_ids=["random", "ucb1"],
        seed=42,
        horizon=50,
    )
    b = run_policy_comparison(
        world_id="rural_clinic",
        policy_ids=["random", "ucb1"],
        seed=42,
        horizon=50,
    )
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        assert ra.policy_id == rb.policy_id
        assert ra.cumulative_reward == rb.cumulative_reward
        assert ra.cumulative_regret == rb.cumulative_regret


def test_orchestrator_runs_multi_policy() -> None:
    results = run_policy_comparison(
        world_id="moviematch",
        policy_ids=["random", "epsilon_greedy", "ucb1"],
        seed=7,
        horizon=30,
    )
    assert len(results) == 3
    policy_ids = {r.policy_id for r in results}
    assert policy_ids == {"random", "epsilon_greedy", "ucb1"}
    for r in results:
        assert r.horizon == 30
        assert r.cumulative_reward >= 0.0
        assert len(r.trace_records) == 30


def test_batch_comparison_multi_seed() -> None:
    results = run_batch_comparison(
        world_id="newsfeed",
        policy_ids=["random", "ucb1"],
        seeds=[1, 2, 3],
        horizon=20,
    )
    assert len(results) == 6


def test_summarize_comparison_runs() -> None:
    results = run_batch_comparison(
        world_id="rural_clinic",
        policy_ids=["random", "ucb1"],
        seeds=[10, 20, 30],
        horizon=40,
    )
    summaries = summarize_comparison_runs(results)
    assert len(summaries) == 2
    summary_ids = {s.policy_id for s in summaries}
    assert summary_ids == {"random", "ucb1"}
    for s in summaries:
        assert s.n_runs == 3
        assert s.mean_reward >= 0.0
        assert s.ci95_half_width >= 0.0


def test_stats_single_run() -> None:
    results = run_policy_comparison(
        world_id="rural_clinic",
        policy_ids=["ucb1"],
        seed=0,
        horizon=10,
    )
    summaries = summarize_comparison_runs(results)
    assert len(summaries) == 1
    s = summaries[0]
    assert s.n_runs == 1
    assert s.std_reward == 0.0
    assert s.ci95_half_width == 0.0
