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
        # Cold start starts at first arm, then round-robins on subsequent calls.
        assert decision.chosen_arm == ARMS[0]

    def test_decide_batch_before_fit_round_robins_arms(self):
        bandit = make_bandit(fitted=False)
        contexts = np.array([make_context() for _ in range(len(ARMS))])
        decisions = bandit.decide_batch(contexts)
        assert [d.chosen_arm for d in decisions] == ARMS

    def test_decision_includes_score_breakdown_for_all_arms(self):
        bandit = make_bandit(fitted=True, policy=PolicyType.UCB1)
        decision = bandit.decide(make_context())
        assert set(decision.score_breakdown.keys()) == {str(a) for a in ARMS}
        chosen = str(decision.chosen_arm)
        assert decision.score_breakdown[chosen].score == pytest.approx(decision.score)
        assert decision.score_breakdown[chosen].mean_estimate is not None
        assert decision.score_breakdown[chosen].confidence_width is not None

    def test_get_model_state_exposes_routed_arm_state(self):
        bandit = make_bandit(fitted=True, policy=PolicyType.LIN_UCB)
        state = bandit.get_model_state(make_context())
        assert state["policy"] == PolicyType.LIN_UCB.value
        assert state["cluster"] >= 0
        assert set(state["arms"].keys()) == {str(a) for a in ARMS}
        assert all("n_obs" in arm_state for arm_state in state["arms"].values())

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

    def test_add_arm_with_custom_gamma(self):
        bandit = make_bandit(fitted=True)
        bandit.add_arm("new_arm", gamma=0.9)
        assert "new_arm" in bandit.arms
        ctx = make_context()
        decision = bandit.decide(ctx)
        assert decision.chosen_arm in bandit.arms

    def test_add_arm_gamma_none_uses_default(self):
        bandit = make_bandit(fitted=True)
        bandit.add_arm("default_gamma_arm", gamma=None)
        assert "default_gamma_arm" in bandit.arms


class TestConfidenceAbstention:
    def test_decide_no_gap_never_abstains(self):
        bandit = make_bandit(fitted=True)
        decision = bandit.decide(make_context(), min_confidence_gap=0.0)
        assert not decision.abstained
        assert decision.chosen_arm is not None

    def test_decide_huge_gap_always_abstains(self):
        bandit = make_bandit(fitted=True)
        decision = bandit.decide(make_context(), min_confidence_gap=1e9)
        assert decision.abstained
        assert decision.chosen_arm is None

    def test_abstained_still_has_all_scores(self):
        bandit = make_bandit(fitted=True)
        decision = bandit.decide(make_context(), min_confidence_gap=1e9)
        assert len(decision.all_scores) == len(ARMS)

    def test_normal_decision_not_abstained_by_default(self):
        bandit = make_bandit(fitted=True)
        decision = bandit.decide(make_context())
        assert not decision.abstained

    def test_cold_start_never_abstains(self):
        bandit = make_bandit(fitted=False)
        decision = bandit.decide(make_context(), min_confidence_gap=1e9)
        # Cold start ignores min_confidence_gap and always returns base arm
        assert not decision.abstained
        assert decision.chosen_arm == ARMS[0]


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


class TestConstrainedBandit:
    def _make_constrained(self, min_pull_rates: dict) -> ClusterBandit:
        return ClusterBandit(
            arms=ARMS,
            n_features=7,
            policy=PolicyType.LIN_UCB,
            n_clusters=3,
            seed=0,
            min_pull_rates=min_pull_rates,
        )

    def test_constrained_bandit_creates_without_error(self):
        bandit = self._make_constrained({1.0: 0.1, 1.1: 0.1})
        assert bandit._constraints._rates is not None

    def test_invalid_rate_raises(self):
        with pytest.raises(ValueError, match="min_pull_rates"):
            self._make_constrained({1.0: 0.0})

    def test_rate_exceeds_one_raises(self):
        with pytest.raises(ValueError, match="min_pull_rates"):
            self._make_constrained({1.0: 1.5})

    def test_sum_exceeds_one_raises(self):
        with pytest.raises(ValueError, match="sum"):
            self._make_constrained({1.0: 0.5, 1.1: 0.6})

    def test_unknown_arm_raises(self):
        with pytest.raises(ValueError, match="unknown arms"):
            self._make_constrained({99.9: 0.1})

    def test_constrained_arm_is_forced_when_under_pulled(self):
        # 1.5 should be chosen frequently since it is constrained at 50% and
        # starts with 0 pulls — the first decision must be for 1.5
        bandit = self._make_constrained({1.5: 0.5})
        rng = np.random.default_rng(0)
        n = 200
        contexts = rng.standard_normal((n, 7))
        decisions = rng.choice(ARMS, size=n)
        rewards = rng.uniform(0, 1, n)
        bandit.fit_offline(contexts, decisions, rewards)

        # After fit, pull 1.5 enough times to see the constraint force it
        pull_counts = {arm: 0 for arm in ARMS}
        for _ in range(200):
            ctx = make_context()
            decision = bandit.decide(ctx)
            arm = decision.chosen_arm
            pull_counts[arm] += 1
            bandit.update(context=ctx, arm=arm, reward=0.5)

        total = sum(pull_counts.values())
        rate_1_5 = pull_counts[1.5] / total
        # With min_pull_rate=0.5, arm 1.5 should be chosen at least 40% of the time
        assert rate_1_5 >= 0.40, f"Expected ≥40% pulls for constrained arm, got {rate_1_5:.2%}"

    def test_remove_arm_cleans_up_constraint(self):
        bandit = self._make_constrained({1.0: 0.1})
        bandit.remove_arm(1.0)
        assert 1.0 not in (bandit._constraints._rates or {})

    def test_unconstrained_bandit_normal_operation(self):
        bandit = make_bandit(fitted=True)
        assert bandit._constraints._rates is None
        decision = bandit.decide(make_context())
        assert decision.chosen_arm in ARMS
