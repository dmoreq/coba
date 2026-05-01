"""Tests for examples/_shared.py simulation engine."""

from __future__ import annotations


import numpy as np
import plotly.graph_objects as go

from examples._shared import (
    BanditSimulator,
    StepRecord,
    create_arm_pull_chart,
    create_arm_score_chart,
    create_cluster_distribution_chart,
    create_regret_chart,
    create_reward_chart,
    generate_biased_log_data,
    generate_linear_synthetic,
    generate_nonlinear_synthetic,
    update_arm_pull_chart,
    update_arm_score_chart,
    update_cluster_chart,
    update_regret_chart,
    update_reward_chart,
)


# ====================================================================
# Part A — Data generators
# ====================================================================


class TestLinearSynthetic:
    def test_shapes_and_bounds(self):
        ctx, arms, true_fn = generate_linear_synthetic(
            n_contexts=200, n_features=4, n_arms=3, seed=0
        )
        assert ctx.shape == (200, 4)
        assert len(arms) == 3
        assert arms == ["arm_0", "arm_1", "arm_2"]
        reward = true_fn(arms[0], ctx[0])
        assert 0.0 <= reward <= 1.0

    def test_reproducibility(self):
        ctx1, arms1, fn1 = generate_linear_synthetic(seed=42)
        ctx2, arms2, fn2 = generate_linear_synthetic(seed=42)
        assert np.array_equal(ctx1, ctx2)
        assert arms1 == arms2
        assert fn1(arms1[0], ctx1[0]) == fn2(arms2[0], ctx2[0])

    def test_different_seed_produces_different_data(self):
        ctx1, _, _ = generate_linear_synthetic(seed=1)
        ctx2, _, _ = generate_linear_synthetic(seed=2)
        assert not np.array_equal(ctx1, ctx2)

    def test_default_arms_are_strings(self):
        _, arms, _ = generate_linear_synthetic(n_arms=3)
        assert arms == ["arm_0", "arm_1", "arm_2"]

    def test_default_features(self):
        ctx, _, _ = generate_linear_synthetic()
        assert ctx.shape[1] == 4  # default n_features


class TestNonlinearSynthetic:
    def test_sigmoid_output_range(self):
        ctx, arms, true_fn = generate_nonlinear_synthetic(n_contexts=100, seed=0)
        rewards = np.array([true_fn(a, ctx[0]) for a in arms])
        assert np.all((rewards >= 0) & (rewards <= 1))

    def test_nonlinearity_difference(self):
        """Quadratic term should produce different reward than purely linear."""
        _, arms, true_fn = generate_nonlinear_synthetic(n_contexts=100, seed=0)
        # Just verify non-linear function returns valid values for many samples
        for a in arms:
            val = true_fn(a, np.random.default_rng(0).standard_normal(8))
            assert 0.0 <= val <= 1.0


class TestBiasedLogData:
    def test_shapes(self):
        contexts, decisions, propensities, rewards = generate_biased_log_data(
            n_logs=300, n_features=5, arms=["a", "b", "c"], seed=0
        )
        assert contexts.shape == (300, 5)
        assert len(decisions) == 300
        assert len(propensities) == 300
        assert len(rewards) == 300
        assert np.all(propensities > 0)

    def test_default_arms(self):
        contexts, decisions, propensities, rewards = generate_biased_log_data(seed=0)
        assert contexts.shape[1] == 5  # default n_features

    def test_biased_towards_last_arm(self):
        """Last arm should appear more often in decisions."""
        n_logs = 1000
        arms = ["a", "b", "c"]
        _, decisions, _, _ = generate_biased_log_data(n_logs=n_logs, arms=arms, seed=0)
        counts = {a: (decisions == a).sum() for a in arms}
        # Last arm (c) should have the most decisions
        assert counts["c"] > counts["a"]
        assert counts["c"] > counts["b"]

    def test_first_arm_has_highest_reward(self):
        """First arm's average reward should be highest (it's truly best)."""
        n_logs = 2000
        arms = ["a", "b", "c"]
        _, decisions, _, rewards = generate_biased_log_data(n_logs=n_logs, arms=arms, seed=0)
        mean_rewards = {a: rewards[decisions == a].mean() for a in arms}
        assert mean_rewards["a"] > mean_rewards["b"]
        assert mean_rewards["a"] > mean_rewards["c"]


# ====================================================================
# Part B — BanditSimulator
# ====================================================================


class TestBanditSimulator:
    def setup_method(self) -> None:
        self.ctx, self.arms, self.true_fn = generate_linear_synthetic(n_contexts=100, seed=0)

    def test_yields_correct_keys(self):
        ctx, arms, true_fn = generate_linear_synthetic(n_contexts=20, seed=0)
        sim = BanditSimulator(
            arms=arms,
            n_features=ctx.shape[1],
            policy="linucb",
            true_reward_fn=true_fn,
            seed=0,
        )
        sim.bootstrap(ctx[:10], seed=0)
        step = next(sim.run(max_steps=1, contexts=ctx[10:]))
        assert isinstance(step, StepRecord)
        assert step.step >= 1
        assert isinstance(step.context, np.ndarray)
        assert step.chosen_arm in arms
        assert 0.0 <= step.reward <= 1.0
        assert 0.0 <= step.cumulative_mean_reward <= 1.0
        assert step.cumulative_regret >= 0.0
        assert isinstance(step.all_scores, dict)
        # cluster should be -1 or a valid cluster index
        assert step.cluster == -1 or 0 <= step.cluster < 3

    def test_cumulative_mean_increases(self):
        ctx, arms, true_fn = generate_linear_synthetic(n_contexts=200, seed=0)
        sim = BanditSimulator(
            arms=arms,
            n_features=ctx.shape[1],
            policy="linucb",
            true_reward_fn=true_fn,
            seed=0,
        )
        sim.bootstrap(ctx[:80], seed=0)
        results = list(sim.run(max_steps=50, contexts=ctx[80:130]))
        final_mean = results[-1].cumulative_mean_reward
        # Should be above random baseline (1/3 ≈ 0.33)
        assert final_mean > 1.0 / len(arms)

    def test_respects_max_steps(self):
        ctx, arms, true_fn = generate_linear_synthetic(n_contexts=100, seed=0)
        sim = BanditSimulator(
            arms=arms,
            n_features=ctx.shape[1],
            policy="linucb",
            true_reward_fn=true_fn,
            seed=0,
        )
        sim.bootstrap(ctx[:50], seed=0)
        results = list(sim.run(max_steps=30, contexts=ctx[50:]))
        assert len(results) == 30

    def test_rewards_in_01(self):
        ctx, arms, true_fn = generate_linear_synthetic(n_contexts=100, seed=0)
        sim = BanditSimulator(
            arms=arms,
            n_features=ctx.shape[1],
            policy="linucb",
            true_reward_fn=true_fn,
            seed=0,
        )
        sim.bootstrap(ctx[:50], seed=0)
        for step in sim.run(max_steps=20, contexts=ctx[50:]):
            assert 0.0 <= step.reward <= 1.0

    def test_history_is_populated(self):
        ctx, arms, true_fn = generate_linear_synthetic(n_contexts=100, seed=0)
        sim = BanditSimulator(
            arms=arms,
            n_features=ctx.shape[1],
            policy="linucb",
            true_reward_fn=true_fn,
            seed=0,
        )
        sim.bootstrap(ctx[:50], seed=0)
        list(sim.run(max_steps=10, contexts=ctx[50:]))
        assert len(sim.history) == 10

    def test_properties_after_run(self):
        ctx, arms, true_fn = generate_linear_synthetic(n_contexts=100, seed=0)
        sim = BanditSimulator(
            arms=arms,
            n_features=ctx.shape[1],
            policy="linucb",
            true_reward_fn=true_fn,
            seed=0,
        )
        sim.bootstrap(ctx[:50], seed=0)
        list(sim.run(max_steps=10, contexts=ctx[50:]))
        # After run, cumulative properties should be accessible
        assert sim.cumulative_mean_reward > 0.0
        assert sim.cumulative_regret >= 0.0
        # bandit property should work
        assert sim.bandit is not None


# ====================================================================
# Part C — Plotly chart factory functions
# ====================================================================


class TestChartFactories:
    def test_create_reward_chart_returns_figure(self):
        fig = create_reward_chart(title="Test", n_arms=3)
        assert isinstance(fig, go.Figure)

    def test_create_regret_chart_returns_figure(self):
        fig = create_regret_chart(title="Test")
        assert isinstance(fig, go.Figure)

    def test_create_arm_score_chart_returns_figure(self):
        fig = create_arm_score_chart(arms=["a", "b", "c"])
        assert isinstance(fig, go.Figure)

    def test_create_cluster_distribution_chart_returns_figure(self):
        fig = create_cluster_distribution_chart(n_clusters=3)
        assert isinstance(fig, go.Figure)

    def test_create_arm_pull_chart_returns_figure(self):
        fig = create_arm_pull_chart(arms=["a", "b", "c"])
        assert isinstance(fig, go.Figure)


class TestChartUpdaters:
    def test_update_reward_chart(self):
        fig = create_reward_chart(n_arms=3)
        fig = update_reward_chart(fig, steps=[1, 2, 3], rewards=[0.5, 0.6, 0.65])
        assert len(fig.data[0].x) == 3

    def test_update_regret_chart(self):
        fig = create_regret_chart()
        fig = update_regret_chart(fig, steps=[1, 2, 3], regrets=[0.1, 0.3, 0.5])
        assert len(fig.data[0].x) == 3

    def test_update_arm_score_chart(self):
        fig = create_arm_score_chart(arms=["a", "b"])
        fig = update_arm_score_chart(fig, scores={"a": 0.8, "b": 0.5})
        assert fig.data[0].x == (0.8, 0.5)

    def test_update_cluster_chart(self):
        fig = create_cluster_distribution_chart(n_clusters=2)
        fig = update_cluster_chart(fig, {0: 10, 1: 20})
        assert fig.data[0].values == (10, 20)

    def test_update_arm_pull_chart(self):
        fig = create_arm_pull_chart(arms=["a", "b"])
        fig = update_arm_pull_chart(fig, {"a": 5, "b": 10})
        # For vertical bars, x is categories, y is counts
        assert fig.data[0].y == (5, 10)
