"""Full lifecycle integration tests for ClusterBandit.

Simulates realistic streams: cold→warm→hot transitions, offline bootstrap
then online updates, drift recovery, and arm churn scenarios.
"""

from __future__ import annotations

import numpy as np
import pytest

from coba import ClusterBandit
from coba.types import PolicyType

ARMS = [1.0, 1.1, 1.2, 1.5]


class TestStreamingLifecycle:
    def test_live_stream_1000_interactions_linucb(self) -> None:
        """1,000-step decide→update loop must complete without crashes,
        all scores must be finite, stats must be bounded."""
        bandit = ClusterBandit(
            arms=ARMS, n_features=3, n_clusters=2, policy=PolicyType.LIN_UCB, seed=0
        )
        rng = np.random.default_rng(0)
        n = 1000
        for i in range(n):
            ctx = rng.standard_normal(3)
            decision = bandit.decide(ctx)
            assert decision.chosen_arm is not None
            assert decision.chosen_arm in bandit.arms
            reward = float(rng.beta(a=2, b=5))  # Beta-distributed rewards
            bandit.update(context=ctx, arm=decision.chosen_arm, reward=reward)
            if i % 200 == 0 and i > 0:
                for s in bandit.get_stats():
                    assert np.isfinite(s.mean_reward), f"Non-finite mean for {s.arm} at step {i}"
        stats = bandit.get_stats()
        assert all(s.n_pulls > 0 for s in stats)

    def test_cold_start_warm_up(self) -> None:
        """Bandit transitions from cold to fitted automatically through updates."""
        # Use n_clusters larger than our update count so auto-bootstrap doesn't
        # fire prematurely — we want to see the cold-start path first.
        bandit = ClusterBandit(arms=ARMS, n_features=3, n_clusters=10, seed=0)
        rng = np.random.default_rng(0)

        # Cold start — round-robin through all arms before auto-bootstrap fires
        cold_start_arms: list = []
        for i in range(3):
            ctx = rng.standard_normal(3)
            decision = bandit.decide(ctx)
            assert decision.chosen_arm in ARMS
            cold_start_arms.append(decision.chosen_arm)
            bandit.update(context=ctx, arm=decision.chosen_arm, reward=float(rng.uniform(0, 1)))
        # Round-robin means first 3 decisions hit arms[0], arms[1], arms[2]
        assert cold_start_arms == list(ARMS[:3])

        # Continue updating past the auto-bootstrap threshold
        for i in range(20):
            ctx = rng.standard_normal(3)
            decision = bandit.decide(ctx)
            bandit.update(context=ctx, arm=decision.chosen_arm, reward=float(rng.uniform(0, 1)))

        # After auto-bootstrap, bandit should be fitted
        assert bandit.is_fitted

    def test_offline_bootstrap_then_online_update(self) -> None:
        """fit_offline(500 logs) → decide→update(200 steps) → scores remain finite."""
        rng = np.random.default_rng(0)
        bandit = ClusterBandit(
            arms=ARMS, n_features=4, n_clusters=3, policy=PolicyType.LIN_TS, seed=0
        )

        # Offline bootstrap
        n_offline = 500
        bandit.fit_offline(
            rng.standard_normal((n_offline, 4)),
            rng.choice(ARMS, size=n_offline),
            rng.uniform(0, 1, n_offline),
        )
        assert bandit.is_fitted

        # Online streaming
        for i in range(200):
            ctx = rng.standard_normal(4)
            decision = bandit.decide(ctx)
            bandit.update(context=ctx, arm=decision.chosen_arm, reward=float(rng.uniform(0, 1)))
            if i % 50 == 0:
                scores = bandit.score_all(rng.standard_normal(4))
                assert all(np.isfinite(v) for v in scores.values())

    @pytest.mark.parametrize(
        "policy",
        [
            PolicyType.LIN_UCB,
            PolicyType.THOMPSON,
            PolicyType.UCB1,
            PolicyType.LOGISTIC_UCB,
            PolicyType.SOFTMAX,
        ],
    )
    def test_policy_lifecycle(self, policy: PolicyType) -> None:
        """Each policy can complete fit→decide→update→get_stats."""
        bandit = ClusterBandit(arms=ARMS, n_features=3, n_clusters=2, policy=policy, seed=0)
        rng = np.random.default_rng(0)
        bandit.fit_offline(
            rng.standard_normal((100, 3)),
            rng.choice(ARMS, size=100),
            rng.uniform(0, 1, 100),
        )
        ctx = rng.standard_normal(3)
        decision = bandit.decide(ctx)
        bandit.update(context=ctx, arm=decision.chosen_arm, reward=0.5)
        stats = bandit.get_stats()
        assert len(stats) == len(ARMS)

    def test_drift_recovery_scenario(self) -> None:
        """Train→drift→detect→recover. After reset, bandit continues to function."""
        bandit = ClusterBandit(
            arms=["A", "B"],
            n_features=3,
            n_clusters=2,
            seed=0,
            enable_drift_detection=True,
            drift_delta=0.0,
            drift_lambda=5.0,
        )
        rng = np.random.default_rng(0)
        bandit.fit_offline(
            rng.standard_normal((100, 3)),
            rng.choice(["A", "B"], size=100),
            rng.uniform(0, 1, 100),
        )
        # Trigger drift on arm A
        for _ in range(100):
            bandit.update(context=rng.standard_normal(3), arm="A", reward=1.0)
        # Bandit should still be operable
        decision = bandit.decide(rng.standard_normal(3))
        assert decision.chosen_arm in ["A", "B"]

    def test_arm_churn_scenario(self) -> None:
        """Add 3 new arms over time, remove 2. Bandit stays consistent."""
        rng = np.random.default_rng(0)
        bandit = ClusterBandit(arms=["A", "B"], n_features=3, n_clusters=2, seed=0)
        bandit.fit_offline(
            rng.standard_normal((100, 3)),
            rng.choice(["A", "B"], size=100),
            rng.uniform(0, 1, 100),
        )
        # Add arms
        bandit.add_arm("C", warm_start_from="A")
        bandit.add_arm("D")
        bandit.add_arm("E", warm_start_from="B", gamma=0.9)
        assert len(bandit.arms) == 5

        # Run some decisions
        for _ in range(20):
            ctx = rng.standard_normal(3)
            decision = bandit.decide(ctx)
            bandit.update(context=ctx, arm=decision.chosen_arm, reward=0.5)

        # Remove arms
        bandit.remove_arm("C")
        bandit.remove_arm("D")
        assert len(bandit.arms) == 3

        # Bandit still works
        scores = bandit.score_all(rng.standard_normal(3))
        assert set(scores.keys()) == {"A", "B", "E"}

    def test_top_k_lifecycle(self) -> None:
        """top_k works at all lifecycle stages."""
        rng = np.random.default_rng(0)
        # Cold start
        bandit = ClusterBandit(arms=ARMS, n_features=3, n_clusters=2, seed=0)
        top = bandit.decide_top_k(rng.standard_normal(3), k=3)
        assert len(top) == 3

        # After fitting
        bandit.fit_offline(
            rng.standard_normal((100, 3)),
            rng.choice(ARMS, size=100),
            rng.uniform(0, 1, 100),
        )
        top = bandit.decide_top_k(rng.standard_normal(3), k=4)
        assert len(top) == 4
        assert top[0][1] >= top[1][1]  # sorted descending
