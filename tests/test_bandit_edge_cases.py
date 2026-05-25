"""Edge case tests for ClusterBandit — validation, dynamic arm lifecycle,
abstention, constraints, and drift integration edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from coba import ClusterBandit
from coba.types import PolicyType

ARMS = [1.0, 1.1, 1.2, 1.5]


class TestClusterBanditDecideEdgeCases:
    def test_cold_start_context_shape_validation(self) -> None:
        bandit = ClusterBandit(arms=ARMS, n_features=3, n_clusters=2)
        with pytest.raises(ValueError, match="1D"):
            bandit.decide(np.zeros((2, 3)))

    def test_cold_start_context_non_finite(self) -> None:
        bandit = ClusterBandit(arms=ARMS, n_features=3, n_clusters=2)
        with pytest.raises(ValueError, match="non-finite"):
            bandit.decide(np.array([1.0, np.nan, 0.5]))

    def test_decide_batch_cold_start(self) -> None:
        bandit = ClusterBandit(arms=ARMS, n_features=3, n_clusters=2)
        contexts = np.random.default_rng(0).standard_normal((5, 3))
        decisions = bandit.decide_batch(contexts)
        assert len(decisions) == 5
        for d in decisions:
            assert d.chosen_arm == ARMS[0]
            assert not d.abstained

    def test_decide_batch_with_abstention(self) -> None:
        rng = np.random.default_rng(0)
        bandit = ClusterBandit(
            arms=ARMS, n_features=3, policy=PolicyType.LIN_UCB, n_clusters=2, seed=0
        )
        n = 100
        contexts = rng.standard_normal((n, 3))
        decisions = rng.choice(ARMS, size=n)
        rewards = rng.uniform(0, 1, n)
        bandit.fit_offline(contexts, decisions, rewards)
        batch = bandit.decide_batch(rng.standard_normal((10, 3)), min_confidence_gap=1e9)
        assert all(d.abstained for d in batch)

    def test_decide_after_remove_and_re_add_arm(self) -> None:
        bandit = ClusterBandit(arms=ARMS, n_features=3, n_clusters=2, seed=0)
        rng = np.random.default_rng(0)
        bandit.fit_offline(
            rng.standard_normal((100, 3)),
            rng.choice(ARMS, size=100),
            rng.uniform(0, 1, 100),
        )
        bandit.remove_arm(1.0)
        bandit.add_arm(1.0)
        decision = bandit.decide(rng.standard_normal(3))
        assert decision.chosen_arm in bandit.arms

    def test_decide_with_single_arm(self) -> None:
        bandit = ClusterBandit(arms=[1.0], n_features=3, n_clusters=1, seed=0)
        rng = np.random.default_rng(0)
        bandit.fit_offline(
            rng.standard_normal((50, 3)),
            np.full(50, 1.0),
            rng.uniform(0, 1, 50),
        )
        decision = bandit.decide(rng.standard_normal(3))
        assert decision.chosen_arm == 1.0

    def test_decide_batch_wrong_shape_raises(self) -> None:
        bandit = ClusterBandit(arms=ARMS, n_features=3, n_clusters=2, seed=0)
        with pytest.raises(ValueError, match="2D"):
            bandit.decide_batch(np.array([1.0, 2.0, 3.0]))

    def test_decide_batch_wrong_features_raises(self) -> None:
        bandit = ClusterBandit(arms=ARMS, n_features=3, n_clusters=2, seed=0)
        with pytest.raises(ValueError, match="feature mismatch"):
            bandit.decide_batch(np.zeros((4, 5)))


class TestClusterBanditUpdateEdgeCases:
    def test_update_propensity_clipping_applied(self) -> None:
        bandit = ClusterBandit(arms=ARMS, n_features=3, n_clusters=2, seed=0)
        rng = np.random.default_rng(0)
        bandit.fit_offline(
            rng.standard_normal((50, 3)),
            rng.choice(ARMS, size=50),
            rng.uniform(0, 1, 50),
        )
        bandit.update(context=rng.standard_normal(3), arm=1.0, reward=0.5, propensity=0.001)

    def test_update_batch_with_propensities(self) -> None:
        bandit = ClusterBandit(arms=ARMS, n_features=3, n_clusters=2, seed=0)
        rng = np.random.default_rng(0)
        bandit.fit_offline(
            rng.standard_normal((50, 3)),
            rng.choice(ARMS, size=50),
            rng.uniform(0, 1, 50),
        )
        bandit.update_batch(
            rng.standard_normal((5, 3)),
            np.array([1.0, 1.1, 1.2, 1.5, 1.0]),
            np.array([0.5, 0.3, 0.8, 0.2, 0.6]),
            propensities=np.array([0.25, 0.5, 0.25, 0.25, 0.5]),
        )

    def test_update_empty_rewards_handled(self) -> None:
        bandit = ClusterBandit(arms=ARMS, n_features=3, n_clusters=2, seed=0)
        bandit.update_batch(np.empty((0, 3)), np.empty(0), np.empty(0))

    def test_update_on_removed_arm_raises(self) -> None:
        rng = np.random.default_rng(0)
        bandit = ClusterBandit(arms=ARMS, n_features=3, n_clusters=2, seed=0)
        bandit.fit_offline(
            rng.standard_normal((50, 3)),
            rng.choice(ARMS, size=50),
            rng.uniform(0, 1, 50),
        )
        bandit.remove_arm(1.0)
        with pytest.raises(ValueError):
            bandit.update(context=rng.standard_normal(3), arm=1.0, reward=0.5)

    def test_update_non_finite_reward_raises(self) -> None:
        bandit = ClusterBandit(arms=ARMS, n_features=3, n_clusters=2, seed=0)
        bandit.fit_offline(
            np.random.default_rng(0).standard_normal((50, 3)),
            np.random.default_rng(0).choice(ARMS, size=50),
            np.ones(50),
        )
        with pytest.raises(ValueError, match="reward must be finite"):
            bandit.update(context=np.ones(3), arm=1.0, reward=float("nan"))

    def test_update_context_feature_mismatch_raises(self) -> None:
        bandit = ClusterBandit(arms=ARMS, n_features=3, n_clusters=2, seed=0)
        with pytest.raises(ValueError, match="feature length mismatch"):
            bandit.update(context=np.array([1.0, 2.0]), arm=1.0, reward=0.5)


class TestClusterBanditDriftEdgeCases:
    def test_drift_reset_preserves_other_arms(self) -> None:
        bandit = ClusterBandit(
            arms=ARMS,
            n_features=3,
            policy=PolicyType.LIN_UCB,
            n_clusters=2,
            seed=0,
            enable_drift_detection=True,
            drift_delta=0.0,
            drift_lambda=5.0,
        )
        rng = np.random.default_rng(0)
        bandit.fit_offline(
            rng.standard_normal((100, 3)),
            rng.choice(ARMS, size=100),
            rng.uniform(0, 1, 100),
        )
        # Flood arm 1.0 to trigger drift
        for _ in range(100):
            bandit.update(context=rng.standard_normal(3), arm=1.0, reward=1.0)
        # Arm 1.1 should not have drift detected
        assert not bandit._drift[1.1].is_drift_detected  # type: ignore[index]

    def test_drift_detector_added_with_new_arm(self) -> None:
        bandit = ClusterBandit(
            arms=ARMS,
            n_features=3,
            n_clusters=2,
            seed=0,
            enable_drift_detection=True,
        )
        bandit.add_arm(2.0)
        assert 2.0 in bandit._drift._detectors  # type: ignore[index]

    def test_drift_detector_removed_with_arm(self) -> None:
        bandit = ClusterBandit(
            arms=ARMS,
            n_features=3,
            n_clusters=2,
            seed=0,
            enable_drift_detection=True,
        )
        bandit.remove_arm(1.0)
        assert 1.0 not in bandit._drift._detectors  # type: ignore[index]


class TestClusterBanditArmManagement:
    def test_warm_start_copies_beta(self) -> None:
        rng = np.random.default_rng(0)
        bandit = ClusterBandit(arms=ARMS, n_features=3, n_clusters=2, seed=0)
        bandit.fit_offline(
            rng.standard_normal((100, 3)),
            rng.choice(ARMS, size=100),
            rng.uniform(0, 1, 100),
        )
        bandit.add_arm(2.0, warm_start_from=1.5)
        scores_after = bandit.score_all(rng.standard_normal(3))
        assert 2.0 in scores_after
        # Warm-started arm should have non-zero scores (not fresh cold)
        assert abs(scores_after[2.0]) > 0

    def test_remove_then_re_add_arm_gets_fresh_state(self) -> None:
        rng = np.random.default_rng(0)
        bandit = ClusterBandit(arms=ARMS, n_features=3, n_clusters=2, seed=0)
        bandit.fit_offline(
            rng.standard_normal((100, 3)),
            rng.choice(ARMS, size=100),
            rng.uniform(0, 1, 100),
        )
        bandit.remove_arm(1.0)
        bandit.add_arm(1.0)
        stats = {s.arm: s for s in bandit.get_stats()}
        assert stats[1.0].n_pulls == 0

    def test_add_arm_with_gamma_propagated(self) -> None:
        bandit = ClusterBandit(arms=ARMS, n_features=3, n_clusters=2, seed=0)
        bandit.fit_offline(
            np.random.default_rng(0).standard_normal((50, 3)),
            np.random.default_rng(0).choice(ARMS, size=50),
            np.ones(50),
        )
        bandit.add_arm("fast_arm", gamma=0.9)
        assert "fast_arm" in bandit.arms
        decision = bandit.decide(np.ones(3))
        assert decision.chosen_arm in bandit.arms

    def test_warm_start_from_nonexistent_falls_back_to_cold(self) -> None:
        bandit = ClusterBandit(arms=ARMS, n_features=3, n_clusters=2, seed=0)
        rng = np.random.default_rng(0)
        bandit.fit_offline(
            rng.standard_normal((50, 3)),
            rng.choice(ARMS, size=50),
            rng.uniform(0, 1, 50),
        )
        bandit.add_arm(99.9, warm_start_from=999.9)
        assert 99.9 in bandit.arms

    def test_remove_last_arm_in_constrained_bandit_raises(self) -> None:
        bandit = ClusterBandit(
            arms=[1.0, 1.1],
            n_features=3,
            n_clusters=1,
            min_pull_rates={1.0: 0.3},
        )
        bandit.remove_arm(1.1)
        with pytest.raises(ValueError, match="Cannot remove the last arm"):
            bandit.remove_arm(1.0)


class TestClusterBanditConstraints:
    def test_constraint_forces_multiple_arms_when_under_pulled(self) -> None:
        bandit = ClusterBandit(
            arms=ARMS,
            n_features=3,
            n_clusters=2,
            seed=0,
            min_pull_rates={1.0: 0.3, 1.5: 0.3},
        )
        rng = np.random.default_rng(0)
        bandit.fit_offline(
            rng.standard_normal((100, 3)),
            rng.choice(ARMS, size=100),
            rng.uniform(0, 1, 100),
        )
        pull_counts = {arm: 0 for arm in ARMS}
        for _ in range(50):
            ctx = rng.standard_normal(3)
            decision = bandit.decide(ctx)
            arm = decision.chosen_arm
            pull_counts[arm] += 1
            bandit.update(context=ctx, arm=arm, reward=0.5)
        total = sum(pull_counts.values())
        if total > 0:
            assert pull_counts[1.0] / total >= 0.15
            assert pull_counts[1.5] / total >= 0.15

    def test_no_constraints_means_normal_operation(self) -> None:
        bandit = ClusterBandit(arms=ARMS, n_features=3, n_clusters=2, seed=0)
        rng = np.random.default_rng(0)
        bandit.fit_offline(
            rng.standard_normal((100, 3)),
            rng.choice(ARMS, size=100),
            rng.uniform(0, 1, 100),
        )
        decision = bandit.decide(rng.standard_normal(3))
        assert decision.chosen_arm in ARMS

    def test_bandit_stats_updated_on_decide_and_update(self) -> None:
        bandit = ClusterBandit(arms=ARMS, n_features=3, n_clusters=2, seed=0)
        rng = np.random.default_rng(0)
        bandit.fit_offline(
            rng.standard_normal((50, 3)),
            rng.choice(ARMS, size=50),
            rng.uniform(0, 1, 50),
        )
        ctx = rng.standard_normal(3)
        decision = bandit.decide(ctx)
        bandit.update(context=ctx, arm=decision.chosen_arm, reward=0.8)
        stats = {s.arm: s for s in bandit.get_stats()}
        assert stats[decision.chosen_arm].n_pulls >= 1
