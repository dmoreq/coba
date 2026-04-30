"""Tests for NeuralLinear backbone, arm model, and ClusterBandit integration."""

import numpy as np

from coba import ClusterBandit
from coba.policies.neural_linear import NeuralLinearArmModel, NeuralLinearBackbone
from coba.types import PolicyType

N_FEATURES = 6
ARMS = ["A", "B", "C"]


def make_backbone(retrain_freq: int = 50) -> NeuralLinearBackbone:
    return NeuralLinearBackbone(
        n_features=N_FEATURES,
        embedding_dim=8,
        hidden_sizes=(16,),
        retrain_freq=retrain_freq,
        seed=0,
    )


def make_arm_model(arm: str = "A", retrain_freq: int = 50) -> NeuralLinearArmModel:
    backbone = make_backbone(retrain_freq=retrain_freq)
    return NeuralLinearArmModel(arm=arm, backbone=backbone, v_sq=1.0, l2_lambda=1.0)


class TestNeuralLinearBackbone:
    def test_initial_state(self):
        bb = make_backbone()
        assert not bb.is_fitted
        assert bb.generation == 0
        assert bb.get_embedding(np.ones(N_FEATURES)) is None

    def test_add_accumulates_buffer(self):
        bb = make_backbone(retrain_freq=100)
        rng = np.random.default_rng(0)
        for _ in range(10):
            bb.add(rng.standard_normal(N_FEATURES), "A", 0.5)
        assert len(bb._buffer) == 10

    def test_retrain_triggered_at_freq(self):
        bb = make_backbone(retrain_freq=20)
        rng = np.random.default_rng(1)
        triggered = False
        for i in range(25):
            result = bb.add(rng.standard_normal(N_FEATURES), "A", rng.uniform(0, 1))
            if result:
                triggered = True
        assert triggered
        assert bb.is_fitted

    def test_get_embedding_returns_correct_shape(self):
        bb = make_backbone(retrain_freq=10)
        rng = np.random.default_rng(2)
        for _ in range(15):
            bb.add(rng.standard_normal(N_FEATURES), "A", rng.uniform(0, 1))
        emb = bb.get_embedding(rng.standard_normal(N_FEATURES))
        assert emb is not None
        assert emb.shape == (8,)

    def test_generation_increments_on_retrain(self):
        bb = make_backbone(retrain_freq=10)
        rng = np.random.default_rng(3)
        for _ in range(25):
            bb.add(rng.standard_normal(N_FEATURES), "A", rng.uniform(0, 1))
        assert bb.generation >= 1

    def test_reset_clears_state(self):
        bb = make_backbone(retrain_freq=10)
        rng = np.random.default_rng(4)
        for _ in range(15):
            bb.add(rng.standard_normal(N_FEATURES), "A", 0.5)
        bb.reset()
        assert not bb.is_fitted
        assert bb.generation == 0
        assert len(bb._buffer) == 0

    def test_buffer_maxlen_drops_old_entries(self):
        bb = NeuralLinearBackbone(
            n_features=N_FEATURES,
            embedding_dim=4,
            hidden_sizes=(8,),
            retrain_freq=1000,
            buffer_maxlen=10,
        )
        rng = np.random.default_rng(5)
        for _ in range(20):
            bb.add(rng.standard_normal(N_FEATURES), "A", 0.5)
        assert len(bb._buffer) == 10


class TestNeuralLinearArmModel:
    def test_initial_not_fitted(self):
        m = make_arm_model()
        assert not m.is_fitted

    def test_score_cold_start_returns_inf(self):
        m = make_arm_model()
        score = m.score(np.ones(N_FEATURES))
        assert np.isinf(score)

    def test_update_marks_fitted(self):
        m = make_arm_model(retrain_freq=10)
        rng = np.random.default_rng(0)
        for _ in range(15):
            m.update(rng.standard_normal(N_FEATURES), reward=0.6)
        assert m.is_fitted

    def test_score_after_retrain_returns_finite(self):
        m = make_arm_model(retrain_freq=10)
        rng = np.random.default_rng(1)
        for _ in range(15):
            m.update(rng.standard_normal(N_FEATURES), reward=0.6)
        score = m.score(rng.standard_normal(N_FEATURES))
        assert np.isfinite(score)

    def test_shared_backbone_across_arms(self):
        bb = make_backbone(retrain_freq=10)
        m_a = NeuralLinearArmModel("A", backbone=bb)
        m_b = NeuralLinearArmModel("B", backbone=bb)
        assert m_a._backbone is m_b._backbone

    def test_reset_resets_lin_ts_only(self):
        m = make_arm_model(retrain_freq=5)
        rng = np.random.default_rng(2)
        for _ in range(10):
            m.update(rng.standard_normal(N_FEATURES), 0.7)
        gen_before = m._backbone.generation
        m.reset()
        assert not m.is_fitted
        assert m._backbone.generation == gen_before  # backbone unchanged

    def test_batch_update(self):
        m = make_arm_model(retrain_freq=20)
        rng = np.random.default_rng(3)
        x_batch = rng.standard_normal((25, N_FEATURES))
        y = rng.uniform(0, 1, 25)
        m.update_batch(x_batch, y)
        assert m.is_fitted


class TestClusterBanditNeuralLinear:
    def _make_bandit(self, retrain_freq: int = 50) -> ClusterBandit:
        return ClusterBandit(
            arms=ARMS,
            n_features=N_FEATURES,
            policy=PolicyType.NEURAL_LINEAR,
            n_clusters=2,
            neural_embedding_dim=8,
            neural_hidden_sizes=(16,),
            neural_retrain_freq=retrain_freq,
            seed=0,
        )

    def test_bandit_creates(self):
        bandit = self._make_bandit()
        assert bandit._router._neural_backbones is not None

    def test_cold_start_returns_first_arm(self):
        bandit = self._make_bandit()
        decision = bandit.decide(np.ones(N_FEATURES))
        # Cold start — backbone not fitted, inf scores, first arm returned
        assert decision.chosen_arm in ARMS

    def test_fit_and_decide(self):
        bandit = self._make_bandit(retrain_freq=50)
        rng = np.random.default_rng(0)
        n = 200
        bandit.fit_offline(
            contexts=rng.standard_normal((n, N_FEATURES)),
            decisions=rng.choice(ARMS, size=n),
            rewards=rng.uniform(0, 1, n),
        )
        decision = bandit.decide(rng.standard_normal(N_FEATURES))
        assert decision.chosen_arm in ARMS

    def test_online_update_loop(self):
        bandit = self._make_bandit(retrain_freq=30)
        rng = np.random.default_rng(1)
        bandit.fit_offline(
            contexts=rng.standard_normal((100, N_FEATURES)),
            decisions=rng.choice(ARMS, size=100),
            rewards=rng.uniform(0, 1, 100),
        )
        for _ in range(50):
            ctx = rng.standard_normal(N_FEATURES)
            decision = bandit.decide(ctx)
            bandit.update(context=ctx, arm=decision.chosen_arm, reward=0.5)

    def test_all_arms_reachable_after_training(self):
        bandit = self._make_bandit(retrain_freq=30)
        rng = np.random.default_rng(2)
        n = 150
        bandit.fit_offline(
            contexts=rng.standard_normal((n, N_FEATURES)),
            decisions=rng.choice(ARMS, size=n),
            rewards=rng.uniform(0, 1, n),
        )
        scores = bandit.score_all(rng.standard_normal(N_FEATURES))
        assert set(scores.keys()) == set(ARMS)
        # All scores should be finite after training (backbone fitted)
        assert all(np.isfinite(v) for v in scores.values())
