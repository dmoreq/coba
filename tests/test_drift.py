"""Tests for PageHinkleyDetector and ClusterBandit drift-detection integration."""

import numpy as np
import pytest

from coba import ClusterBandit, PageHinkleyDetector
from coba.types import PolicyType

# ---------------------------------------------------------------------------
# PageHinkleyDetector unit tests
# ---------------------------------------------------------------------------


class TestPageHinkleyDetectorInit:
    def test_default_params(self):
        d = PageHinkleyDetector()
        assert d.delta == 0.005
        assert d.lambda_ == 50.0
        assert d.alpha == 0.999

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError, match="delta"):
            PageHinkleyDetector(delta=-0.1)

    def test_invalid_lambda_raises(self):
        with pytest.raises(ValueError, match="lambda_"):
            PageHinkleyDetector(lambda_=0.0)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            PageHinkleyDetector(alpha=1.5)

    def test_initial_state(self):
        d = PageHinkleyDetector()
        assert d.n_samples == 0
        assert not d.is_drift_detected
        assert d.reference_mean == 0.0


class TestPageHinkleyDetectorUpdate:
    def test_stable_stream_no_drift(self):
        d = PageHinkleyDetector(delta=0.01, lambda_=30.0)
        rng = np.random.default_rng(0)
        for _ in range(500):
            val = rng.normal(0.5, 0.1)
            d.update(val)
        assert not d.is_drift_detected

    def test_detects_upward_shift(self):
        d = PageHinkleyDetector(delta=0.01, lambda_=20.0)
        rng = np.random.default_rng(1)
        # Stable phase: mean ≈ 0.2
        for _ in range(200):
            d.update(rng.normal(0.2, 0.05))
        # Shift: mean jumps to 0.8
        detected = False
        for _ in range(300):
            if d.update(rng.normal(0.8, 0.05)):
                detected = True
                break
        assert detected, "Expected upward drift to be detected"

    def test_detects_downward_shift(self):
        d = PageHinkleyDetector(delta=0.01, lambda_=20.0)
        rng = np.random.default_rng(2)
        # Stable phase: mean ≈ 0.8
        for _ in range(200):
            d.update(rng.normal(0.8, 0.05))
        # Shift: mean drops to 0.2
        detected = False
        for _ in range(300):
            if d.update(rng.normal(0.2, 0.05)):
                detected = True
                break
        assert detected, "Expected downward drift to be detected"

    def test_n_samples_increments(self):
        d = PageHinkleyDetector()
        for i in range(10):
            d.update(0.5)
        assert d.n_samples == 10

    def test_reference_mean_tracks_signal(self):
        d = PageHinkleyDetector(alpha=0.9)
        for _ in range(100):
            d.update(1.0)
        # After many 1.0 observations, mean should be close to 1.0
        assert abs(d.reference_mean - 1.0) < 0.1

    def test_update_returns_bool(self):
        d = PageHinkleyDetector()
        result = d.update(0.5)
        assert isinstance(result, bool)


class TestPageHinkleyDetectorReset:
    def test_reset_clears_cumulative_sums(self):
        d = PageHinkleyDetector(delta=0.01, lambda_=20.0)
        rng = np.random.default_rng(3)
        for _ in range(200):
            d.update(rng.normal(0.2, 0.05))
        for _ in range(300):
            d.update(rng.normal(0.8, 0.05))
            if d.is_drift_detected:
                break
        # Reset without clearing mean
        mean_before = d.reference_mean
        d.reset()
        assert not d.is_drift_detected
        # Mean is preserved
        assert abs(d.reference_mean - mean_before) < 1e-10

    def test_full_reset_clears_everything(self):
        d = PageHinkleyDetector()
        for _ in range(100):
            d.update(0.7)
        d.full_reset()
        assert d.n_samples == 0
        assert d.reference_mean == 0.0
        assert not d.is_drift_detected

    def test_no_drift_after_reset(self):
        d = PageHinkleyDetector(delta=0.01, lambda_=20.0)
        rng = np.random.default_rng(4)
        for _ in range(200):
            d.update(rng.normal(0.2, 0.05))
        # Trigger drift
        for _ in range(300):
            d.update(rng.normal(0.8, 0.05))
            if d.is_drift_detected:
                break
        d.reset()
        # Stable observations at new mean should not re-trigger immediately
        assert not d.is_drift_detected


# ---------------------------------------------------------------------------
# ClusterBandit drift integration tests
# ---------------------------------------------------------------------------


ARMS = ["A", "B", "C"]


def make_bandit_with_drift(**kwargs) -> ClusterBandit:
    return ClusterBandit(
        arms=ARMS,
        n_features=3,
        policy=PolicyType.LIN_UCB,
        n_clusters=2,
        seed=0,
        enable_drift_detection=True,
        **kwargs,
    )


class TestClusterBanditDriftIntegration:
    def test_bandit_without_drift_detection(self):
        bandit = ClusterBandit(
            arms=ARMS, n_features=3, policy=PolicyType.LIN_UCB, n_clusters=2, seed=0
        )
        assert bandit._drift._detectors is None

    def test_bandit_with_drift_detection_has_detectors(self):
        bandit = make_bandit_with_drift()
        assert bandit._drift._detectors is not None
        assert set(bandit._drift._detectors.keys()) == set(ARMS)

    def test_add_arm_registers_detector(self):
        bandit = make_bandit_with_drift()
        bandit.add_arm("D")
        assert "D" in bandit._drift._detectors  # type: ignore[index]

    def test_remove_arm_unregisters_detector(self):
        bandit = make_bandit_with_drift()
        bandit.remove_arm("C")
        assert "C" not in bandit._drift._detectors  # type: ignore[index]

    def test_drift_triggers_arm_reset(self):
        # Use very low lambda_ so drift fires quickly
        bandit = make_bandit_with_drift(drift_delta=0.0, drift_lambda=5.0)
        rng = np.random.default_rng(0)
        n = 200
        contexts = rng.standard_normal((n, 3))
        decisions = rng.choice(ARMS, size=n)
        rewards = rng.uniform(0, 1, n)
        bandit.fit_offline(contexts, decisions, rewards)

        # Send a flood of updates to arm "A" — should trigger reset without error
        for _ in range(30):
            ctx = rng.standard_normal(3)
            bandit.update(context=ctx, arm="A", reward=rng.uniform(0, 1))
        # Bandit should still be operable after drift reset
        decision = bandit.decide(rng.standard_normal(3))
        assert decision.chosen_arm in ARMS

    def test_normal_operation_unaffected_without_drift(self):
        bandit = make_bandit_with_drift(drift_delta=1.0, drift_lambda=1e6)
        rng = np.random.default_rng(5)
        n = 100
        contexts = rng.standard_normal((n, 3))
        decisions = rng.choice(ARMS, size=n)
        rewards = rng.uniform(0, 1, n)
        bandit.fit_offline(contexts, decisions, rewards)

        for _ in range(20):
            ctx = rng.standard_normal(3)
            bandit.update(context=ctx, arm="A", reward=0.5)

        assert not bandit._drift["A"].is_drift_detected  # type: ignore[index]
