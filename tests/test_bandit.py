"""Tests for the ClusterBandit facade."""

import numpy as np
import pytest

from coba import BanditDecision, ClusterBandit
from coba.types import PolicyType

ARMS = [1.0, 1.1, 1.2, 1.5]


def make_context() -> np.ndarray:
    return np.array([50.0, 10.0, 100.0, 5.0, 8.0, 1.0, 5.0])


def make_bandit(fitted: bool = False, policy: PolicyType = PolicyType.LIN_UCB) -> ClusterBandit:
    bandit = ClusterBandit(arms=ARMS, n_features=7, policy=policy, n_clusters=3, seed=0)
    if fitted:
        rng = np.random.default_rng(0)
        n = 200
        contexts = rng.standard_normal((n, 7))
        decisions = rng.choice(ARMS, size=n)
        rewards = rng.uniform(0, 1, n)
        bandit.fit_offline(contexts, decisions, rewards)
    return bandit


class TestClusterBanditDecide:
    def test_decide_before_fit_returns_base_arm(self):
        bandit = make_bandit(fitted=False)
        ctx = make_context()
        decision = bandit.decide(ctx)
        assert isinstance(decision, BanditDecision)
        # Cold start: returns first arm
        assert decision.chosen_arm == ARMS[0]

    def test_decide_after_fit_returns_valid_arm(self):
        bandit = make_bandit(fitted=True)
        ctx = make_context()
        decision = bandit.decide(ctx)
        assert decision.chosen_arm in ARMS

    def test_decide_all_scores_populated(self):
        bandit = make_bandit(fitted=True)
        ctx = make_context()
        decision = bandit.decide(ctx)
        assert len(decision.all_scores) == len(ARMS)

    def test_chosen_arm_has_highest_score(self):
        bandit = make_bandit(fitted=True)
        ctx = make_context()
        decision = bandit.decide(ctx)
        max_score = max(decision.all_scores.values())
        assert decision.score == pytest.approx(max_score)


class TestClusterBanditUpdate:
    def test_update_single(self):
        bandit = make_bandit(fitted=True)
        ctx = make_context()
        bandit.update(
            context=ctx,
            arm=1.0,
            reward=0.7,
            propensity=0.5,
        )  # Should not raise

    def test_update_increases_pull_count(self):
        bandit = make_bandit(fitted=True)
        ctx = make_context()
        bandit.update(context=ctx, arm=1.0, reward=0.7)
        stats = {s.arm: s for s in bandit.get_stats()}
        assert stats[1.0].n_pulls >= 1

    def test_update_batch(self):
        bandit = make_bandit(fitted=True)
        contexts = np.array([make_context() for _ in ARMS])
        arms = np.array(ARMS)
        rewards = np.full(len(ARMS), 0.5)
        bandit.update_batch(contexts, arms, rewards)  # Should not raise

    def test_update_batch_empty_list(self):
        bandit = make_bandit(fitted=True)
        bandit.update_batch(np.array([]), np.array([]), np.array([]))  # Should not raise


class TestClusterBanditFitFromLogs:
    def test_fit_marks_fitted(self):
        bandit = make_bandit(fitted=False)
        rng = np.random.default_rng(1)
        contexts = rng.standard_normal((100, 7))
        decisions = rng.choice(ARMS, 100)
        rewards = rng.uniform(0, 1, 100)
        bandit.fit_offline(contexts, decisions, rewards)
        assert bandit.is_fitted

    def test_fit_with_propensities(self):
        bandit = make_bandit(fitted=False)
        rng = np.random.default_rng(2)
        n = 100
        contexts = rng.standard_normal((n, 7))
        decisions = rng.choice(ARMS, n)
        rewards = rng.uniform(0, 1, n)
        propensities = np.full(n, 0.25)
        bandit.fit_offline(contexts, decisions, rewards, propensities=propensities)
        assert bandit.is_fitted

    def test_fit_returns_self(self):
        bandit = make_bandit(fitted=False)
        rng = np.random.default_rng(3)
        n = 100
        result = bandit.fit_offline(
            rng.standard_normal((n, 7)),
            rng.choice(ARMS, n),
            rng.uniform(0, 1, n),
        )
        assert result is bandit


class TestClusterBanditArmManagement:
    def test_add_arm(self):
        bandit = make_bandit(fitted=True)
        bandit.add_arm(2.0)
        assert 2.0 in bandit.arms

    def test_add_arm_warm_start(self):
        bandit = make_bandit(fitted=True)
        bandit.add_arm(2.0, warm_start_from=1.5)
        assert 2.0 in bandit.arms
        ctx = make_context()
        # Should be able to decide with new arm
        decision = bandit.decide(ctx)
        assert decision.chosen_arm in bandit.arms

    def test_remove_arm(self):
        bandit = make_bandit(fitted=True)
        bandit.remove_arm(1.0)
        assert 1.0 not in bandit.arms

    def test_stats_updated_after_arm_add(self):
        bandit = make_bandit(fitted=True)
        bandit.add_arm(2.0)
        stats_arms = {s.arm for s in bandit.get_stats()}
        assert 2.0 in stats_arms

    def test_stats_removed_after_arm_remove(self):
        bandit = make_bandit(fitted=True)
        bandit.remove_arm(1.0)
        stats_arms = {s.arm for s in bandit.get_stats()}
        assert 1.0 not in stats_arms


class TestClusterBanditTopK:
    def test_top_k_before_fit_returns_first_k_arms(self):
        bandit = make_bandit(fitted=False)
        result = bandit.decide_top_k(make_context(), k=2)
        assert len(result) == 2
        assert result[0][0] == ARMS[0]
        assert result[1][0] == ARMS[1]

    def test_top_k_after_fit_returns_k_arms(self):
        bandit = make_bandit(fitted=True)
        result = bandit.decide_top_k(make_context(), k=2)
        assert len(result) == 2
        arms_returned = [arm for arm, _ in result]
        for arm in arms_returned:
            assert arm in ARMS

    def test_top_k_scores_descending(self):
        bandit = make_bandit(fitted=True)
        result = bandit.decide_top_k(make_context(), k=3)
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_clamped_to_n_arms(self):
        bandit = make_bandit(fitted=True)
        result = bandit.decide_top_k(make_context(), k=100)
        assert len(result) == len(ARMS)

    def test_top_k_first_matches_decide(self):
        bandit = make_bandit(fitted=True)
        ctx = make_context()
        top1_arm = bandit.decide_top_k(ctx, k=1)[0][0]
        decided_arm = bandit.decide(ctx).chosen_arm
        assert top1_arm == decided_arm

    def test_top_k_zero_raises(self):
        bandit = make_bandit(fitted=True)
        with pytest.raises(ValueError, match="k must be at least 1"):
            bandit.decide_top_k(make_context(), k=0)


class TestClusterBanditMonitoring:
    def test_get_stats_returns_all_arms(self):
        bandit = make_bandit(fitted=True)
        stats = bandit.get_stats()
        assert len(stats) == len(ARMS)

    def test_get_cluster_assignment_returns_valid_cluster(self):
        bandit = make_bandit(fitted=True)
        ctx = make_context()
        cluster = bandit.get_cluster_assignment(ctx)
        assert 0 <= cluster < bandit.n_clusters

    def test_is_fitted_property(self):
        bandit = make_bandit(fitted=False)
        assert not bandit.is_fitted
        bandit = make_bandit(fitted=True)
        assert bandit.is_fitted

    def test_score_all_returns_all_arms(self):
        bandit = make_bandit(fitted=True)
        scores = bandit.score_all(make_context())
        assert set(scores.keys()) == set(ARMS)

    def test_offline_eval_methods_return_expected_types(self):
        bandit = make_bandit(fitted=True)
        rng = np.random.default_rng(5)
        n = 60
        contexts = rng.standard_normal((n, 7))
        decisions = rng.choice(ARMS, n)
        rewards = rng.uniform(0, 1, n)
        propensities = np.full(n, 0.25)
        reward_estimates = rng.uniform(0, 1, n)
        rs = bandit.evaluate_rejection_sampling(contexts, decisions, rewards)
        dr = bandit.evaluate_doubly_robust(
            contexts, decisions, rewards, propensities, reward_estimates
        )
        ncis = bandit.evaluate_ncis(
            policy_scores=np.ones(n),
            logging_scores=np.full(n, 0.5),
            rewards=rewards,
        )
        assert rs.method == "rejection_sampling"
        assert dr.method == "doubly_robust"
        assert ncis.method == "ncis"

    @pytest.mark.parametrize(
        "policy",
        [
            PolicyType.LIN_UCB,
            PolicyType.LIN_TS,
            PolicyType.THOMPSON,
            PolicyType.UCB1,
            PolicyType.LOGISTIC_UCB,
            PolicyType.LOGISTIC_TS,
        ],
    )
    def test_all_policies_end_to_end(self, policy):
        bandit = make_bandit(fitted=True, policy=policy)
        ctx = make_context()
        decision = bandit.decide(ctx)
        assert decision.chosen_arm in ARMS

        bandit.update(context=ctx, arm=decision.chosen_arm, reward=0.5)

    def test_update_invalid_propensity_raises(self):
        bandit = make_bandit(fitted=True)
        with pytest.raises(ValueError, match="propensity"):
            bandit.update(context=make_context(), arm=1.0, reward=0.5, propensity=0.0)

    def test_decide_invalid_context_shape_raises(self):
        bandit = make_bandit(fitted=True)
        with pytest.raises(ValueError, match="1D"):
            bandit.decide(np.zeros((2, 7)))

    def test_update_batch_mismatched_lengths_raises(self):
        bandit = make_bandit(fitted=True)
        contexts = np.zeros((3, 7))
        arms = np.array([1.0, 1.1])
        rewards = np.array([0.3, 0.4, 0.5])
        with pytest.raises(ValueError, match="same length"):
            bandit.update_batch(contexts, arms, rewards)

    def test_fit_offline_dr_requires_reward_estimates(self):
        bandit = make_bandit(fitted=False)
        rng = np.random.default_rng(4)
        n = 50
        contexts = rng.standard_normal((n, 7))
        decisions = rng.choice(ARMS, n)
        rewards = rng.uniform(0, 1, n)
        propensities = np.full(n, 0.25)
        with pytest.raises(ValueError, match="reward_estimates"):
            bandit.fit_offline(
                contexts=contexts,
                decisions=decisions,
                rewards=rewards,
                propensities=propensities,
                use_dr=True,
                reward_estimates=None,
            )
