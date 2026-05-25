"""Deep mathematical correctness tests for RidgeRegression (Sherman-Morrison).

These tests verify mathematical invariants that a naive Sherman-Morrison
implementation could silently violate. They complement the basic unit tests
in test_policies.py by covering numerical stability, gamma decay correctness,
batch/sequential equivalence, and edge cases.
"""

from __future__ import annotations

import numpy as np

from coba.policies.ridge import RidgeRegression


# ── Mathematical invariants ──────────────────────────────────────────────────


class TestRidgeShermanMorrisonInvariants:
    """Sherman-Morrison rank-1 update produces the same result as direct matrix inversion."""

    def test_sm_matches_direct_inverse_with_gamma_decay(self) -> None:
        """SM updates with gamma < 1.0 must produce the same beta as computing
        the exponentially-decayed A matrix and Xty accumulator, then inverting."""
        n_features = 4
        l2_lambda = 2.0
        gamma = 0.95
        n_obs = 50
        rng = np.random.default_rng(42)

        x_mat = rng.standard_normal((n_obs, n_features))
        y_vec = rng.uniform(0, 1, n_obs)

        # SM incremental path
        ridge = RidgeRegression(n_features, l2_lambda=l2_lambda, gamma=gamma)
        for xi, yi in zip(x_mat, y_vec):
            ridge.update(xi, float(yi))

        # Direct decayed computation:
        # A_t = l2_lambda * I + sum_{i=1..t} gamma^{t-i} * x_i x_i^T
        a_direct = l2_lambda * np.eye(n_features)
        xty_direct = np.zeros(n_features)
        for i in range(n_obs):
            xi = x_mat[i]
            yi = y_vec[i]
            # Apply decay to existing accumulators before adding new observation
            a_direct = gamma * a_direct + np.outer(xi, xi)
            xty_direct = gamma * xty_direct + xi * yi
        beta_direct = np.linalg.solve(a_direct, xty_direct)

        np.testing.assert_array_almost_equal(ridge.beta, beta_direct, decimal=4)

    def test_batch_vs_sequential_identical(self) -> None:
        """update_batch(X, y) must produce exactly the same beta as calling
        update() sequentially for each row."""
        rng = np.random.default_rng(0)
        n_features = 5
        n_obs = 30
        x_mat = rng.standard_normal((n_obs, n_features))
        y_vec = rng.uniform(0, 1, n_obs)

        ridge_batch = RidgeRegression(n_features)
        ridge_batch.update_batch(x_mat, y_vec)

        ridge_seq = RidgeRegression(n_features)
        for xi, yi in zip(x_mat, y_vec):
            ridge_seq.update(xi, float(yi))

        np.testing.assert_array_almost_equal(ridge_batch.beta, ridge_seq.beta, decimal=10)
        np.testing.assert_array_almost_equal(ridge_batch.A_inv, ridge_seq.A_inv, decimal=10)
        np.testing.assert_array_almost_equal(ridge_batch.Xty, ridge_seq.Xty, decimal=10)

    def test_weighted_batch_vs_weighted_sequential(self) -> None:
        """update_batch(X, y, weights=w) must produce same result as calling
        update(x_i, y_i, weight=w_i) for each row."""
        rng = np.random.default_rng(1)
        n_features = 3
        n_obs = 20
        x_mat = rng.standard_normal((n_obs, n_features))
        y_vec = rng.uniform(0, 1, n_obs)
        weights = rng.uniform(0.5, 3.0, n_obs)

        ridge_batch = RidgeRegression(n_features)
        ridge_batch.update_batch(x_mat, y_vec, weights=weights)

        ridge_seq = RidgeRegression(n_features)
        for xi, yi, wi in zip(x_mat, y_vec, weights):
            ridge_seq.update(xi, float(yi), weight=float(wi))

        np.testing.assert_array_almost_equal(ridge_batch.beta, ridge_seq.beta, decimal=8)

    def test_beta_identical_to_solve_when_gamma_is_one(self) -> None:
        """When gamma=1.0 (stationary), beta must equal np.linalg.solve(A, Xty)."""
        rng = np.random.default_rng(2)
        n_features = 4
        l2_lambda = 3.0
        n_obs = 40
        x_mat = rng.standard_normal((n_obs, n_features))
        y_vec = rng.uniform(0, 1, n_obs)

        ridge = RidgeRegression(n_features, l2_lambda=l2_lambda, gamma=1.0)
        ridge.update_batch(x_mat, y_vec)

        a_direct = l2_lambda * np.eye(n_features) + x_mat.T @ x_mat
        xty_direct = x_mat.T @ y_vec
        beta_direct = np.linalg.solve(a_direct, xty_direct)

        np.testing.assert_array_almost_equal(ridge.beta, beta_direct, decimal=6)

    def test_a_matrix_equals_inverse_of_a_inv(self) -> None:
        """After any number of updates, ridge.A (which calls np.linalg.inv)
        must equal the actual inverse of ridge.A_inv."""
        rng = np.random.default_rng(3)
        n_features = 3
        ridge = RidgeRegression(n_features)

        for _ in range(25):
            x = rng.standard_normal(n_features)
            ridge.update(x, float(rng.uniform(0, 1)))

        a_from_property = ridge.A
        a_via_inv = np.linalg.inv(ridge.A_inv)
        np.testing.assert_array_almost_equal(a_from_property, a_via_inv, decimal=8)


# ── Numerical stability ─────────────────────────────────────────────────────


class TestRidgeNumericalStability:
    """Tests that Sherman-Morrison doesn't degrade under heavy use."""

    def test_many_updates_no_nan_or_inf(self) -> None:
        """10,000 sequential updates with random data must not produce NaN or inf."""
        rng = np.random.default_rng(0)
        n_features = 5
        ridge = RidgeRegression(n_features, l2_lambda=1.0, gamma=0.999)

        for _ in range(10_000):
            x = rng.standard_normal(n_features)
            y = float(rng.uniform(0, 1))
            ridge.update(x, y)
            assert np.all(np.isfinite(ridge.A_inv)), "A_inv contains non-finite values"
            assert np.all(np.isfinite(ridge.beta)), "beta contains non-finite values"

        assert ridge.n_obs == 10_000

    def test_denom_never_explodes_with_small_values(self) -> None:
        """After many gamma-downdates, the denominator in SM must always stay above
        1e-10 and never become negative — verified by checking A_inv remains PSD."""
        rng = np.random.default_rng(1)
        n_features = 4
        # gamma=0.9 is aggressive — A_inv grows by 1/0.9 ≈ 1.11 each step.
        # After 500 updates A_inv ≈ 1.11^500 ≈ 10^21 → this stresses the damping.
        ridge = RidgeRegression(n_features, l2_lambda=1.0, gamma=0.9)

        for _ in range(500):
            x = rng.standard_normal(n_features) * 0.01  # small contexts to stress denominator
            y = float(rng.uniform(0, 1))
            ridge.update(x, y)

        # A_inv must remain symmetric positive definite
        eigvals = np.linalg.eigvalsh(ridge.A_inv)
        assert np.all(eigvals > 0), f"Smallest eigenvalue: {eigvals.min()}"
        assert np.all(np.isfinite(eigvals))

    def test_gamma_decay_limits_history_influence(self) -> None:
        """With gamma < 1.0, very old observations must have effectively zero
        influence on beta. After 100 steps of gamma=0.9, the first observation
        is weighted at 0.9^99 ≈ 3e-5 — practically zero."""
        n_features = 3
        ridge = RidgeRegression(n_features, l2_lambda=1.0, gamma=0.9)

        # First observation: large signal in one direction
        x_first = np.array([10.0, 0.0, 0.0])
        y_first = 1.0
        ridge.update(x_first, y_first)
        beta_after_first = ridge.beta.copy()

        # Drown the first observation in 200 random updates
        rng = np.random.default_rng(5)
        for _ in range(200):
            x = rng.standard_normal(n_features) * 0.1
            y = float(rng.uniform(0, 1))
            ridge.update(x, y)

        # Beta should have moved significantly away from the first-observation value
        assert not np.allclose(ridge.beta, beta_after_first, atol=1e-2)


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestRidgeEdgeCases:
    """Tests for edge cases that could cause silent failures."""

    def test_zero_feature_vector(self) -> None:
        """x = [0, ..., 0] — the zero vector must not corrupt A_inv."""
        n_features = 3
        ridge = RidgeRegression(n_features)
        a_inv_before = ridge.A_inv.copy()

        ridge.update(np.zeros(n_features), y=0.5)

        # With x=0, the SM denominator is 1 + 0 = 1, numerator is outer(0, 0) = 0.
        # A_inv must be unchanged.
        np.testing.assert_array_almost_equal(ridge.A_inv, a_inv_before)

    def test_negative_reward(self) -> None:
        """Rewards outside [0, 1] must be handled correctly without corruption."""
        ridge = RidgeRegression(n_features=2)
        ridge.update(np.array([1.0, 0.5]), y=-5.0)
        ridge.update(np.array([0.3, 0.8]), y=10.0)
        assert np.all(np.isfinite(ridge.beta))

    def test_single_feature(self) -> None:
        """n_features=1 edge case — SM formula still works."""
        ridge = RidgeRegression(n_features=1, l2_lambda=1.0)
        ridge.update(np.array([2.0]), y=0.8)
        ridge.update(np.array([-1.0]), y=0.3)
        assert np.isfinite(ridge.beta[0])
        assert ridge.n_obs == 2

    def test_reset_then_reuse_produces_fresh_results(self) -> None:
        """After reset(), continuing to update must produce results independent
        of the pre-reset state."""
        ridge = RidgeRegression(n_features=2)
        ridge.update(np.array([1.0, 0.0]), y=0.9)
        ridge.update_batch(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0.9, 0.1]))

        # Record state
        beta_before_reset = ridge.beta.copy()

        ridge.reset()
        assert ridge.n_obs == 0
        assert np.allclose(ridge.beta, np.zeros(2))

        # Now fit the same data again
        ridge.update(np.array([1.0, 0.0]), y=0.9)
        ridge.update_batch(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0.9, 0.1]))

        np.testing.assert_array_almost_equal(ridge.beta, beta_before_reset)

    def test_reset_with_different_lambda(self) -> None:
        """reset(l2_lambda=10.0) must change the A_inv prior."""
        ridge = RidgeRegression(n_features=2, l2_lambda=1.0)
        ridge.update(np.array([1.0, 0.0]), y=0.8)
        ridge.reset(l2_lambda=100.0)
        # A_inv should now be (1/100) * I
        expected = (1.0 / 100.0) * np.eye(2)
        np.testing.assert_array_almost_equal(ridge.A_inv, expected)

    def test_weight_zero_is_noop(self) -> None:
        """weight=0 update must not change any internal state beyond n_obs."""
        ridge = RidgeRegression(n_features=2)
        a_inv_before = ridge.A_inv.copy()
        xty_before = ridge.Xty.copy()

        ridge.update(np.array([1.0, 0.5]), y=0.9, weight=0.0)

        np.testing.assert_array_almost_equal(ridge.A_inv, a_inv_before)
        np.testing.assert_array_almost_equal(ridge.Xty, xty_before)
        # n_obs still increments — it counts observations, not effective samples
        assert ridge.n_obs == 1

    def test_large_lambda_makes_a_inv_small(self) -> None:
        """λ=1e6 must produce A_inv ≈ 1e-6 * I."""
        ridge = RidgeRegression(n_features=3, l2_lambda=1e6)
        expected = (1.0 / 1e6) * np.eye(3)
        np.testing.assert_array_almost_equal(ridge.A_inv, expected)

    def test_batch_update_empty_arrays(self) -> None:
        """Empty batch update is a no-op."""
        ridge = RidgeRegression(n_features=2)
        ridge.update_batch(np.empty((0, 2)), np.empty(0), np.empty(0))
        assert ridge.n_obs == 0
        assert np.allclose(ridge.beta, np.zeros(2))

    def test_very_small_lambda_produces_large_a_inv(self) -> None:
        """λ=1e-8 → A_inv ≈ 1e8 * I. Updates must still be numerically stable."""
        ridge = RidgeRegression(n_features=2, l2_lambda=1e-8)
        ridge.update(np.array([1.0, 0.0]), y=0.5)
        assert np.all(np.isfinite(ridge.beta))
        assert np.all(np.isfinite(ridge.A_inv))
