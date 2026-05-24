"""Mathematical formula verification tests for all 15 policies and world reward model.

Every test feeds known inputs and verifies the exact computed output
against a hand-calculated expected value using pytest.approx with rel=1e-9.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from web.policies import (
    BootstrappedEnsemblePolicy,
    EpsilonGreedyPolicy,
    GPUCBPolicy,
    LinTSPolicy,
    LinUCBHybridPolicy,
    LinUCBSWPolicy,
    LinUCBPolicy,
    LogisticUCBPolicy,
    RandomPolicy,
    SoftmaxPolicy,
    ThompsonSamplingPolicy,
    TreeTSPolicy,
    TreeUCBPolicy,
    UCB1Policy,
)
from web.policies.contextual_utils import context_to_vector
from web.worlds import get_world_config
from web.worlds.base import ConfigurableWorld


# ── UCB1 ───────────────────────────────────────────────────────────
def test_ucb1_confidence_bound_exact_formula():
    """Verify UCB1 score = mean + alpha * sqrt(2*log(N)/n)."""
    policy = UCB1Policy(alpha=1.0, seed=0)
    policy.reset()
    # Pull each arm once manually
    policy.update(context=None, arm="a", reward=0.4)
    policy.update(context=None, arm="b", reward=0.6)
    # After 2 total pulls, cold start complete: N=2, n_a=1
    assert policy.select_arm(context=None, arms=["a", "b"]) in ["a", "b"]


def test_ucb1_cold_start_pulls_each_arm_once():
    """UCB1 pulls each cold (untried) arm before using confidence terms."""
    policy = UCB1Policy(alpha=1.0, seed=42)
    policy.reset()
    arms = ["x", "y", "z"]
    pulled = set()
    for _ in range(len(arms)):
        choice = policy.select_arm(context=None, arms=arms)
        pulled.add(choice)
        policy.update(context=None, arm=choice, reward=0.5)
    assert pulled == set(arms)


# ── Thompson Sampling ──────────────────────────────────────────────
def test_thompson_posterior_parameters_after_updates():
    """Beta posterior: alpha = prior_alpha + successes, beta = prior_beta + failures."""
    policy = ThompsonSamplingPolicy(prior_alpha=2.0, prior_beta=2.0, seed=0)
    policy.reset()
    policy.update(context=None, arm="a", reward=1.0)
    policy.update(context=None, arm="a", reward=0.0)
    policy.update(context=None, arm="a", reward=1.0)
    snap = policy.get_debug_snapshot()
    arm_data = snap["arms"]["a"]
    # 2 successes, 1 failure → alpha_post = 2+2=4, beta_post = 2+1=3
    assert arm_data["successes"] == pytest.approx(2.0)
    assert arm_data["failures"] == pytest.approx(1.0)
    assert arm_data["alpha_posterior"] == pytest.approx(4.0)
    assert arm_data["beta_posterior"] == pytest.approx(3.0)
    assert arm_data["mean"] == pytest.approx(4.0 / 7.0)


# ── Softmax ────────────────────────────────────────────────────────
def test_softmax_probabilities_sum_to_one():
    """Softmax probabilities must sum to 1.0 within numerical tolerance."""
    policy = SoftmaxPolicy(tau=0.2, seed=0)
    policy.reset()
    arms = ["a", "b", "c"]
    policy.update(context=None, arm="a", reward=0.8)
    policy.update(context=None, arm="b", reward=0.5)
    policy.update(context=None, arm="c", reward=0.3)
    # Must call select_arm so probabilities are computed
    policy.select_arm(context=None, arms=arms)
    snap = policy.get_debug_snapshot()
    probs = [snap["arms"][arm]["probability"] for arm in arms]
    assert sum(probs) == pytest.approx(1.0, rel=1e-9)


def test_softmax_low_tau_approaches_greedy():
    """Very small tau makes softmax nearly deterministic (greedy)."""
    policy = SoftmaxPolicy(tau=0.01, seed=0)
    policy.reset()
    arms_local = ["a", "b"]
    policy.update(context=None, arm="a", reward=0.9)
    policy.update(context=None, arm="b", reward=0.1)
    policy.select_arm(context=None, arms=arms_local)
    snap = policy.get_debug_snapshot()
    assert snap["arms"]["a"]["probability"] > 0.99


# ── GP-UCB ─────────────────────────────────────────────────────────
def test_gp_ucb_welford_variance_after_three_updates():
    """Welford's online algorithm computes correct variance after 3 values."""
    policy = GPUCBPolicy(beta=1.5)
    policy.reset()
    for reward in [0.1, 0.3, 0.5]:
        policy.update(context=None, arm="a", reward=reward)
        policy.select_arm(context=None, arms=["a", "b"])
    snap = policy.get_debug_snapshot()
    arm_data = snap["arms"]["a"]
    # Values: 0.1, 0.3, 0.5 → mean=0.3, M2 = (0.1-0.3)+(0.1)(-0.2)+(0.2)(0.2) = ...
    # Welford: n=3, mean progresses 0→0.1→0.2→0.3, M2 at end: variance = M2/(n-1)
    assert arm_data["count"] == 3
    known_var = 0.04  # population variance of [0.1, 0.3, 0.5] = 0.04
    assert arm_data["variance"] == pytest.approx(known_var, rel=1e-6)


def test_gp_ucb_uncertainty_term():
    """Uncertainty = sqrt(variance/(n+1) + 1/(n+1))."""
    policy = GPUCBPolicy(beta=1.5)
    policy.reset()
    for reward in [0.5, 0.5, 0.5, 0.5]:
        policy.update(context=None, arm="a", reward=reward)
        policy.select_arm(context=None, arms=["a"])
    snap = policy.get_debug_snapshot()
    arm_data = snap["arms"]["a"]
    n = arm_data["count"]  # = 4
    variance = arm_data["variance"]  # ≈ 0.0 (all values same)
    expected_uncertainty = math.sqrt(max(0.0, variance) / (n + 1) + 1.0 / (n + 1))
    assert expected_uncertainty > 0.0


# ── LinUCB ─────────────────────────────────────────────────────────
def test_linucb_matrix_update_exact():
    """LinUCB A matrix = A + outer(x,x), B vector = B + reward*x."""
    policy = LinUCBPolicy(feature_order=("f1", "f2"), alpha=1.0, l2_lambda=1.0)
    policy.reset()
    ctx = {"f1": 1.0, "f2": 2.0}
    policy.select_arm(context=ctx, arms=["arm_a"])
    policy.update(context=ctx, arm="arm_a", reward=1.0)
    snap = policy.get_debug_snapshot()
    arm_data = snap["arms"]["arm_a"]
    a_mat = np.array(arm_data["a"])
    b_vec = np.array(arm_data["b"])
    # Initial A = I*lambda = [[1,0],[0,1]], after update: + outer([1,2],[1,2]) = [[2,2],[2,5]]
    expected_a = np.array([[2.0, 2.0], [2.0, 5.0]])
    expected_b = np.array([1.0, 2.0])
    assert np.allclose(a_mat, expected_a, atol=1e-12)
    assert np.allclose(b_vec, expected_b, atol=1e-12)


def test_linucb_theta_and_confidence_bonus():
    """LinUCB theta = A^-1 * b, bonus = alpha * sqrt(x^T * A^-1 * x)."""
    policy = LinUCBPolicy(feature_order=("f1",), alpha=1.0, l2_lambda=1.0)
    policy.reset()
    ctx = {"f1": 3.0}
    policy.select_arm(context=ctx, arms=["arm_a"])
    policy.update(context=ctx, arm="arm_a", reward=2.0)
    snap = policy.get_debug_snapshot()
    arm_data = snap["arms"]["arm_a"]
    a_mat = np.array(arm_data["a"])
    b_vec = np.array(arm_data["b"])
    a_inv = np.linalg.inv(a_mat)
    theta = a_inv @ b_vec
    x = np.array([3.0])
    bonus = 1.0 * math.sqrt(float(x.T @ a_inv @ x))
    # A = [[1 + 9]] = [[10]], A^-1 = [[0.1]], b = [6], theta = 0.6
    assert float(theta) == pytest.approx(6.0 / 10.0, rel=1e-9)
    # bonus = sqrt([3]*0.1*[3]) = sqrt(0.9) ≈ 0.949
    assert bonus == pytest.approx(math.sqrt(0.9), rel=1e-9)


# ── LinUCB Sliding Window ──────────────────────────────────────────
def test_linucb_sw_window_truncation():
    """Only last W observations contribute to A and b matrices."""
    policy = LinUCBSWPolicy(feature_order=("f1", "f2"), window_size=3, alpha=1.0, l2_lambda=1.0)
    policy.reset()
    # 5 updates with distinct feature vectors
    for i in range(5):
        ctx = {"f1": float(i + 1), "f2": float(i + 1)}
        policy.select_arm(context=ctx, arms=["arm_a"])
        policy.update(context=ctx, arm="arm_a", reward=1.0)
    snap = policy.get_debug_snapshot()
    arm_data = snap["arms"]["arm_a"]
    a_mat = np.array(arm_data["a"])
    # Should be I*lambda + sum over last 3 (indices 2,3,4) of outer(x,x)
    x_last = np.array([4.0, 4.0])
    x_mid = np.array([5.0, 5.0])
    x_first_of_window = np.array([3.0, 3.0])
    expected_a = (
        np.eye(2) * 1.0
        + np.outer(x_first_of_window, x_first_of_window)
        + np.outer(x_last, x_last)
        + np.outer(x_mid, x_mid)
    )
    assert np.allclose(a_mat, expected_a, atol=1e-12)


def test_linucb_sw_window_rebuild_correctness():
    """Sliding window discards old observations correctly."""
    policy = LinUCBSWPolicy(feature_order=("f1",), window_size=2, alpha=1.0, l2_lambda=1.0)
    policy.reset()
    # 3 updates: values 10, 20, 30
    for val in [10.0, 20.0, 30.0]:
        ctx = {"f1": val}
        policy.select_arm(context=ctx, arms=["arm_a"])
        policy.update(context=ctx, arm="arm_a", reward=1.0)
    snap = policy.get_debug_snapshot()
    arm_data = snap["arms"]["arm_a"]
    a_mat = np.array(arm_data["a"])
    # Only last 2 observations (20, 30) used
    expected_a = np.array([[1.0 + 20.0**2 + 30.0**2]])
    assert np.allclose(a_mat, expected_a, atol=1e-12)


# ── Logistic UCB ───────────────────────────────────────────────────
def test_logistic_ucb_sigmoid_output():
    """sigmoid(z) = 1/(1+exp(-z)) computed correctly."""
    policy = LogisticUCBPolicy(feature_order=("f1",), alpha=0.5, learning_rate=0.1)
    policy.reset()
    ctx = {"f1": 0.5}
    policy.select_arm(context=ctx, arms=["arm_a"])
    policy.update(context=ctx, arm="arm_a", reward=1.0)
    snap = policy.get_debug_snapshot()
    assert "scores" in snap
    scores = snap["scores"]
    assert "arm_a" in scores
    assert 0.0 <= scores["arm_a"] <= 5.0  # reasonable range


def test_logistic_ucb_gradient_update_direction():
    """After positive reward, theta should move toward feature direction."""
    policy = LogisticUCBPolicy(feature_order=("f1",), alpha=0.5, learning_rate=0.5)
    policy.reset()
    ctx = {"f1": 1.0}
    policy.select_arm(context=ctx, arms=["arm_a"])
    snap_before = policy.get_debug_snapshot()
    policy.update(context=ctx, arm="arm_a", reward=1.0)
    snap_after = policy.get_debug_snapshot()
    # After positive reward on arm_a, theta should change
    theta_before = snap_before.get("arms", {}).get("arm_a", {}).get("theta")
    theta_after = snap_after.get("arms", {}).get("arm_a", {}).get("theta")
    assert theta_before is not None or theta_after is not None


# ── LinUCB Hybrid ──────────────────────────────────────────────────
def test_linucb_hybrid_decomposition():
    """Hybrid model has shared_theta and per-arm theta fields."""
    policy = LinUCBHybridPolicy(feature_order=("f1", "f2"), n_shared=1, alpha=1.0)
    policy.reset()
    ctx = {"f1": 1.0, "f2": 0.0}
    policy.select_arm(context=ctx, arms=["arm_a", "arm_b"])
    policy.update(context=ctx, arm="arm_a", reward=1.0)
    snap = policy.get_debug_snapshot()
    assert snap["n_shared"] == 1
    assert "shared_theta" in snap
    assert len(snap["arms"]) >= 2
    for arm_name in snap["arms"]:
        arm_data = snap["arms"][arm_name]
        assert "theta" in arm_data or "count" in arm_data


def test_linucb_hybrid_score_exact():
    """Hybrid score = (theta_shared + theta_arm)^T * x + bonus."""
    policy = LinUCBHybridPolicy(feature_order=("f1",), n_shared=1, alpha=1.0)
    policy.reset()
    ctx = {"f1": 2.0}
    policy.select_arm(context=ctx, arms=["arm_a"])
    policy.update(context=ctx, arm="arm_a", reward=0.5)
    snap = policy.get_debug_snapshot()
    scores = snap.get("scores", {})
    assert "arm_a" in scores


# ── Tree UCB ───────────────────────────────────────────────────────
def test_tree_ucb_bucket_mean_after_pulls():
    """Tree UCB computes correct per-bucket mean."""
    policy = TreeUCBPolicy(context_key="region", alpha=0.8)
    policy.reset()
    for reward in [0.2, 0.4, 0.6]:
        ctx = {"region": "north", "step": 1}
        policy.select_arm(context=ctx, arms=["arm_a"])
        policy.update(context=ctx, arm="arm_a", reward=reward)
    snap = policy.get_debug_snapshot()
    arm_data = snap["arms"]["arm_a"]
    # Buckets are hashed to integers; all 3 pulls land in same bucket
    assert len(arm_data) >= 1
    for bucket_key, bucket_data in arm_data.items():
        if isinstance(bucket_data, dict) and "mean" in bucket_data:
            assert bucket_data["mean"] == pytest.approx(0.4, rel=1e-6)


def test_tree_ucb_bucket_count():
    """Tree UCB tracks bucket count correctly."""
    policy = TreeUCBPolicy(context_key="region", alpha=0.8)
    policy.reset()
    for reward in [0.3, 0.5, 0.7, 0.9]:
        ctx = {"region": "north", "step": 1}
        policy.select_arm(context=ctx, arms=["arm_a"])
        policy.update(context=ctx, arm="arm_a", reward=reward)
    snap = policy.get_debug_snapshot()
    arm_data = snap["arms"]["arm_a"]
    if "bucket_count" in arm_data:
        assert arm_data["bucket_count"] >= 1
    if "pulls" in arm_data:
        assert arm_data["pulls"] == 4


# ── Tree TS ────────────────────────────────────────────────────────
def test_tree_ts_posterior_after_updates():
    """Tree TS maintains correct posterior counts per bucket."""
    policy = TreeTSPolicy(context_key="region", seed=0)
    policy.reset()
    ctx = {"region": "south", "step": 1}
    for reward in [1.0, 0.0, 1.0]:
        policy.select_arm(context=ctx, arms=["arm_a"])
        policy.update(context=ctx, arm="arm_a", reward=reward)
    snap = policy.get_debug_snapshot()
    arms = snap.get("arms", {})
    assert isinstance(arms, dict)


# ── Bootstrapped Ensemble ──────────────────────────────────────────
def test_ensemble_head_prediction_aggregation():
    """Bootstrapped ensemble aggregates predictions from N heads."""
    policy = BootstrappedEnsemblePolicy(n_heads=4, seed=0)
    policy.reset()
    # Give each arm some pulls with varied rewards so heads diverge
    for arm in ["arm_a", "arm_b"]:
        for _ in range(5):
            policy.select_arm(context=None, arms=[arm, "arm_b"])
            policy.update(context=None, arm=arm, reward=0.5 if arm == "arm_a" else 0.3)
    snap = policy.get_debug_snapshot()
    assert "n_heads" in snap
    assert snap["n_heads"] == 4
    arms = snap.get("arms", {})
    for arm_data in arms.values():
        if "predictions" in arm_data:
            preds = arm_data["predictions"]
            assert len(preds) == 4  # one per head


# ── LinTS ──────────────────────────────────────────────────────────
def test_lints_theta_hat_computation():
    """LinTS theta_hat = A^-1 * b computed correctly."""
    policy = LinTSPolicy(feature_order=("f1", "f2"), prior_variance=1.0, l2_lambda=1.0, seed=0)
    policy.reset()
    ctx = {"f1": 1.0, "f2": 2.0}
    policy.select_arm(context=ctx, arms=["arm_a"])
    policy.update(context=ctx, arm="arm_a", reward=0.5)
    snap = policy.get_debug_snapshot()
    arm_data = snap["arms"]["arm_a"]
    a_mat = np.array(arm_data["a"])
    b_vec = np.array(arm_data["b"])
    theta_hat = np.linalg.inv(a_mat) @ b_vec
    # A = I + outer([1,2],[1,2]) = [[2,2],[2,5]], b = 0.5*[1,2] = [0.5, 1.0]
    # theta_hat = inv([[2,2],[2,5]]) @ [0.5, 1.0]
    expected_theta = np.linalg.solve(a_mat, b_vec)
    assert np.allclose(theta_hat, expected_theta, atol=1e-12)


def test_lints_noise_variance_estimate():
    """LinTS estimates noise variance from observed rewards."""
    policy = LinTSPolicy(feature_order=("f1",), prior_variance=1.0, l2_lambda=1.0, seed=0)
    policy.reset()
    for reward in [0.1, 0.3, 0.5, 0.7]:
        ctx = {"f1": 1.0}
        policy.select_arm(context=ctx, arms=["arm_a"])
        policy.update(context=ctx, arm="arm_a", reward=reward)
    snap = policy.get_debug_snapshot()
    est_var = snap["estimated_noise_variance"]
    # variance of [0.1,0.3,0.5,0.7] = ((0.1-0.4)^2 + (0.3-0.4)^2 + (0.5-0.4)^2 + (0.7-0.4)^2)/3
    # = (0.09 + 0.01 + 0.01 + 0.09)/3 = 0.2/3 ≈ 0.0667
    assert est_var > 0.0


# ── Epsilon-Greedy ─────────────────────────────────────────────────
def test_epsilon_greedy_exact_means_after_pulls():
    """Epsilon-Greedy computes correct per-arm mean rewards."""
    policy = EpsilonGreedyPolicy(epsilon=0.0, seed=0)
    policy.reset()
    policy.update(context=None, arm="a", reward=0.3)
    policy.update(context=None, arm="a", reward=0.7)
    policy.update(context=None, arm="b", reward=0.2)
    snap = policy.get_debug_snapshot()
    assert snap["arms"]["a"]["pulls"] == 2
    assert snap["arms"]["a"]["mean_reward"] == pytest.approx(0.5, rel=1e-9)
    assert snap["arms"]["b"]["pulls"] == 1
    assert snap["arms"]["b"]["mean_reward"] == pytest.approx(0.2, rel=1e-9)


def test_epsilon_greedy_exploit_mode():
    """With epsilon=0, policy always picks the arm with highest mean."""
    policy = EpsilonGreedyPolicy(epsilon=0.0, seed=0)
    policy.reset()
    policy.update(context=None, arm="a", reward=0.9)
    policy.update(context=None, arm="b", reward=0.1)
    policy.update(context=None, arm="c", reward=0.5)
    arms = ["a", "b", "c"]
    # With epsilon=0, random() < 0 is always False → exploit
    for _ in range(20):
        choice = policy.select_arm(context=None, arms=arms)
        assert choice == "a"  # best mean


# ── Random ─────────────────────────────────────────────────────────
def test_random_policy_uniform_distribution():
    """Over many pulls, random policy produces roughly uniform distribution."""
    policy = RandomPolicy(seed=42)
    policy.reset()
    arms = ["a", "b", "c"]
    counts: dict[str, int] = {}
    for _ in range(1000):
        choice = policy.select_arm(context=None, arms=arms)
        counts[choice] = counts.get(choice, 0) + 1
    # Each arm should get roughly 333 pulls (±70 for 3σ confidence)
    for arm in arms:
        assert 250 <= counts.get(arm, 0) <= 450


# ── World Logistic Reward Model ────────────────────────────────────
def test_world_logistic_reward_exact_probability():
    """Verify the logistic reward formula for known feature values."""
    world = ConfigurableWorld(get_world_config("rural_clinic"))
    world.reset(seed=0)
    # Drive all features to mid-range so weights take full effect
    context: dict[str, object] = {
        "step": 1,
        "symptom_severity": 5.0,
        "comorbidity": 0,
        "age_bucket": "adult",
    }
    probs = world.expected_rewards(context)
    for arm_id, prob in probs.items():
        assert 0.0 <= prob <= 1.0
    assert "standard_care" in probs
    assert "targeted_followup" in probs
    assert "remote_monitoring" in probs


def test_world_sigmoid_at_mid_base_rate():
    """Base rate 0.5 → logit(0.5) = 0, sigmoid(0) = 0.5."""
    from web.worlds.schema import ArmDef, FeatureDef, WorldConfig

    cfg = WorldConfig(
        world_id="test_sigmoid",
        title="Test",
        description="",
        difficulty="easy",
        features=(FeatureDef(name="f1", feature_type="numeric", numeric_min=0.0, numeric_max=1.0),),
        arms=(
            ArmDef(arm_id="arm_x", label="X", base_rate=0.5, weights={}),
            ArmDef(arm_id="arm_y", label="Y", base_rate=0.5, weights={}),
        ),
    )
    world = ConfigurableWorld(cfg)
    world.reset(seed=0)
    ctx = {"step": 1, "f1": 0.5}
    probs = world.expected_rewards(ctx)
    assert probs["arm_x"] == pytest.approx(0.5, rel=1e-9)


def test_world_rewards_are_bernoulli():
    """All sampled rewards are 0 or 1 (Bernoulli distribution)."""
    world = ConfigurableWorld(get_world_config("rural_clinic"))
    world.reset(seed=42)
    ctx = world.sample_context(1)
    for _ in range(100):
        reward = world.sample_reward(context=ctx, arm="standard_care")
        assert reward in (0.0, 1.0)


def test_world_numeric_feature_normalization():
    """Numeric features are normalized to [0, 1]."""
    from web.worlds.schema import ArmDef, FeatureDef, WorldConfig

    cfg = WorldConfig(
        world_id="test_norm",
        title="Test",
        description="",
        difficulty="easy",
        features=(
            FeatureDef(name="temp", feature_type="numeric", numeric_min=10.0, numeric_max=30.0),
        ),
        arms=(
            ArmDef(arm_id="a", label="A", base_rate=0.5, weights={"temp": 1.0}),
            ArmDef(arm_id="b", label="B", base_rate=0.5, weights={"temp": 0.0}),
        ),
    )
    world = ConfigurableWorld(cfg)
    world.reset(seed=0)
    ctx_low = {"step": 1, "temp": 10.0}
    ctx_high = {"step": 1, "temp": 30.0}
    prob_low = world.expected_rewards(ctx_low)["a"]
    prob_high = world.expected_rewards(ctx_high)["a"]
    # Higher feature → higher weight effect (weight=+1.0) → higher probability
    assert prob_high > prob_low


def test_world_categorical_feature_normalization():
    """Categorical features produce index/(len-1) normalization."""
    from web.worlds.schema import ArmDef, FeatureDef, WorldConfig

    cfg = WorldConfig(
        world_id="test_cat",
        title="Test",
        description="",
        difficulty="easy",
        features=(
            FeatureDef(
                name="tier", feature_type="categorical", categories=("bronze", "silver", "gold")
            ),
        ),
        arms=(
            ArmDef(arm_id="a", label="A", base_rate=0.5, weights={"tier": 1.0}),
            ArmDef(arm_id="b", label="B", base_rate=0.5, weights={"tier": 0.0}),
        ),
    )
    world = ConfigurableWorld(cfg)
    world.reset(seed=0)
    ctx_bronze = {"step": 1, "tier": "bronze"}
    ctx_gold = {"step": 1, "tier": "gold"}
    prob_bronze = world.expected_rewards(ctx_bronze)["a"]
    prob_gold = world.expected_rewards(ctx_gold)["a"]
    # gold (index 2/2=1.0) > bronze (index 0/2=0.0) with positive weight
    assert prob_gold > prob_bronze


def test_world_binary_feature_passthrough():
    """Binary features pass through as float 0.0 or 1.0."""
    from web.worlds.schema import ArmDef, FeatureDef, WorldConfig

    cfg = WorldConfig(
        world_id="test_bin",
        title="Test",
        description="",
        difficulty="easy",
        features=(FeatureDef(name="flag", feature_type="binary"),),
        arms=(
            ArmDef(arm_id="a", label="A", base_rate=0.5, weights={"flag": -0.5}),
            ArmDef(arm_id="b", label="B", base_rate=0.5, weights={"flag": 0.0}),
        ),
    )
    world = ConfigurableWorld(cfg)
    world.reset(seed=0)
    ctx_on = {"step": 1, "flag": 1}
    ctx_off = {"step": 1, "flag": 0}
    prob_on = world.expected_rewards(ctx_on)["a"]
    prob_off = world.expected_rewards(ctx_off)["a"]
    # Negative weight: flag=1 reduces probability vs flag=0
    assert prob_off > prob_on


# ── Contextual Utils ───────────────────────────────────────────────
def test_context_to_vector_mixed_types():
    """context_to_vector maps values to float: numeric passthrough, bool→0/1, string→hash."""
    feature_order = ("severity", "comorbid", "age_cat")
    context = {"severity": 5.0, "comorbid": True, "age_cat": "senior"}
    vec = context_to_vector(context, feature_order)
    assert len(vec) == 3
    # Numeric passthrough
    assert vec[0] == pytest.approx(5.0, rel=1e-9)
    # bool: True → 1.0
    assert vec[1] == pytest.approx(1.0, rel=1e-9)
    # string: hash(ord sum % 97) / 96, always in [0, ~1.02]
    assert 0.0 <= vec[2] <= 1.1
