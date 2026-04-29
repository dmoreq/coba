"""Tests for Logistic Bandits (Laplace Approximation)."""


import numpy as np

from coba.policies.logistic import LogisticTSArmModel, LogisticUCBArmModel, sigmoid


def test_sigmoid():
    """Test stable sigmoid function."""
    assert np.isclose(sigmoid(0.0), 0.5)
    assert np.isclose(sigmoid(100.0), 1.0)
    assert np.isclose(sigmoid(-100.0), 0.0)


def test_logistic_ucb_initialization():
    """Test Logistic UCB init and optimistic score."""
    model = LogisticUCBArmModel(arm="test_arm", n_features=3, alpha=1.0)

    assert model.n_obs == 0
    assert not model.is_fitted

    # Initially w is 0, so logit_mu = 0
    # Variance is x^T H_inv x. H_inv is identity.
    # Score should be sigmoid(alpha * sqrt(norm(x)^2))
    x = np.array([1.0, 0.0, 0.0])
    score = model.score(x)

    assert score > 0.5  # Optimistic exploration


def test_logistic_ts_sampling():
    """Test Logistic TS sampling produces different scores but centered around mu."""
    model = LogisticTSArmModel(arm="test_arm", n_features=2, v_sq=1.0)
    x = np.array([1.0, 1.0])

    scores = [model.score(x) for _ in range(100)]

    # Should not all be exactly the same
    assert len(set(scores)) > 1
    # Since w=0 initially, logit_mu=0, samples should be centered around 0.5
    assert 0.1 < np.mean(scores) < 0.9


def test_logistic_learning():
    """Test that the model learns a simple binary pattern."""
    model = LogisticUCBArmModel(arm="test_arm", n_features=2, alpha=0.1)

    # Pattern: Feature 1 positive -> reward 1. Feature 2 positive -> reward 0.
    x_mat = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    y_vec = np.array([1.0, 1.0, 0.0, 0.0, 1.0])

    model.update_batch(x_mat, y_vec)

    assert model.is_fitted
    assert model.n_obs == 5

    # Predict
    prob_1 = model.score(np.array([1.0, 0.0]))
    prob_0 = model.score(np.array([0.0, 1.0]))

    assert prob_1 > prob_0


def test_logistic_gamma_decay():
    """Test that gamma < 1.0 decays the inverse Hessian correctly."""
    model1 = LogisticUCBArmModel(arm="test_arm", n_features=2, gamma=1.0)
    model2 = LogisticUCBArmModel(arm="test_arm", n_features=2, gamma=0.5)

    x = np.array([1.0, 0.0])

    # Update both
    model1.update(x, 1.0)
    model2.update(x, 1.0)

    # Model 2 should have larger H_inv elements because it was divided by 0.5 before the update
    # Note: Before update, H_inv is 1.0.
    # For model 1: H_inv becomes 1.0. Then Sherman Morrison downdates it.
    # For model 2: H_inv becomes 2.0. Then Sherman Morrison downdates it.
    # So model2.model.H_inv[1, 1] should be 2.0 (untouched by x=[1, 0])
    assert np.isclose(model1.model.H_inv[1, 1], 1.0)
    assert np.isclose(model2.model.H_inv[1, 1], 2.0)


def test_logistic_reset():
    """Test that reset clears state."""
    model = LogisticUCBArmModel(arm="test_arm", n_features=2)
    model.update(np.array([1.0, 0.0]), 1.0)

    assert model.is_fitted
    assert model.n_obs == 1

    model.reset()
    assert not model.is_fitted
    assert model.n_obs == 0
    assert np.allclose(model.model.w, 0.0)


def test_logistic_ts_update_batch() -> None:
    """LogisticTS update_batch should update state and mark model as fitted."""
    model = LogisticTSArmModel(arm="ts_arm", n_features=2, v_sq=1.0)
    x_mat = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y_vec = np.array([1.0, 0.0, 1.0])

    model.update_batch(x_mat, y_vec)

    assert model.is_fitted
    assert model.n_obs == 3


def test_logistic_ts_reset() -> None:
    """LogisticTS reset should clear all learned state."""
    model = LogisticTSArmModel(arm="ts_arm", n_features=2)
    model.update(np.array([1.0, 0.0]), 1.0)

    assert model.is_fitted
    assert model.n_obs == 1

    model.reset()

    assert not model.is_fitted
    assert model.n_obs == 0
    # Weight vector should be back to zero prior
    assert np.allclose(model.model.w, 0.0)


def test_logistic_ucb_clone_independence() -> None:
    """Cloning a LogisticUCB model should produce an independent copy."""
    model = LogisticUCBArmModel(arm="clone_arm", n_features=2, alpha=1.0)
    model.update(np.array([1.0, 0.0]), 1.0)

    cloned = model.clone()
    # Modifying the original should not change the clone
    model.update(np.array([0.0, 1.0]), 0.0)

    assert cloned.n_obs == 1  # clone still sees only the first update
    assert model.n_obs == 2
