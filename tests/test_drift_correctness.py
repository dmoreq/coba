"""Statistical property tests for PageHinkleyDetector.

These verify that the two-sided Page-Hinkley test has correct false-positive
control, bounded detection latency, and proper reset behavior. They complement
the basic unit tests in test_drift.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from coba.drift import PageHinkleyDetector


class TestPageHinkleyStatisticalProperties:
    """Statistical behavior of the PH test under controlled conditions."""

    def test_no_false_positives_on_stationary_stream(self) -> None:
        """10,000 i.i.d. samples from the same distribution must produce
        zero drift detections under sensible default parameters."""
        detector = PageHinkleyDetector(delta=0.005, lambda_=50.0, alpha=0.999)
        rng = np.random.default_rng(0)

        detections = 0
        for _ in range(10_000):
            if detector.update(float(rng.normal(0.5, 0.1))):
                detections += 1
                detector.reset()

        assert detections == 0, f"Got {detections} false positives on 10k stationary samples"

    def test_detection_latency_with_known_shift(self) -> None:
        """After a deterministic upward shift from 0.0 to 1.0 with delta=0.01,
        lambda_=20, drift must be detected within ~2*lambda_/delta=4000 samples.
        In practice it should be much faster for a clean step change."""
        detector = PageHinkleyDetector(delta=0.01, lambda_=20.0, alpha=0.999)

        # Establish baseline at 0.0
        for _ in range(500):
            detector.update(0.0)

        # Abrupt shift to 1.0
        detected = False
        delay = 0
        for i in range(2000):
            if detector.update(1.0):
                detected = True
                delay = i + 1
                break

        assert detected, "Expected drift to be detected after step change"
        assert delay < 2000, f"Detection latency {delay} too high"

    def test_higher_delta_reduces_sensitivity(self) -> None:
        """Higher delta must reduce false-positive rate on a noisy stationary stream."""
        rng = np.random.default_rng(1)

        # Low delta: more sensitive
        det_low = PageHinkleyDetector(delta=0.0, lambda_=100.0)
        detections_low = 0
        for _ in range(5000):
            if det_low.update(float(rng.normal(0.5, 0.15))):
                detections_low += 1
                det_low.reset()

        # Reset RNG to the same state for fair comparison
        rng = np.random.default_rng(1)

        # High delta: less sensitive
        det_high = PageHinkleyDetector(delta=0.05, lambda_=100.0)
        detections_high = 0
        for _ in range(5000):
            if det_high.update(float(rng.normal(0.5, 0.15))):
                detections_high += 1
                det_high.reset()

        assert (
            detections_low >= detections_high
        ), f"delta=0.0: {detections_low}, delta=0.05: {detections_high}"

    def test_two_sided_both_directions_detectable(self) -> None:
        """First shift up, then shift down — both must be detected."""
        detector = PageHinkleyDetector(delta=0.005, lambda_=20.0)

        # Baseline at 0.5
        rng = np.random.default_rng(2)
        for _ in range(200):
            detector.update(float(rng.normal(0.5, 0.03)))

        # Shift up to 1.0
        up_detected = False
        for _ in range(500):
            if detector.update(float(rng.normal(1.0, 0.03))):
                up_detected = True
                break
        assert up_detected, "Upward shift not detected"
        detector.reset()

        # Shift down to 0.0
        down_detected = False
        for _ in range(500):
            if detector.update(float(rng.normal(0.0, 0.03))):
                down_detected = True
                break
        assert down_detected, "Downward shift not detected"

    def test_alpha_controls_mean_adaptation_rate(self) -> None:
        """alpha=0.9 (faster adaptation) vs alpha=0.999 (slower).
        After a shift, the mean of the fast detector must reach the new value sooner."""
        # Fast adapter
        d_fast = PageHinkleyDetector(alpha=0.9, delta=0.005, lambda_=500.0)
        # Slow adapter
        d_slow = PageHinkleyDetector(alpha=0.999, delta=0.005, lambda_=500.0)

        # Baseline at 0.0
        for _ in range(200):
            d_fast.update(0.0)
            d_slow.update(0.0)

        # Both see a shift to 1.0
        for i in range(100):
            d_fast.update(1.0)
            d_slow.update(1.0)
            if d_fast.reference_mean > d_slow.reference_mean:
                return  # d_fast adapted faster — pass

        fast_mean = d_fast.reference_mean
        slow_mean = d_slow.reference_mean
        assert fast_mean > slow_mean, f"After 100 steps: fast={fast_mean:.4f}, slow={slow_mean:.4f}"


class TestPageHinkleyEdgeCases:
    """Edge cases for the PH detector."""

    def test_single_observation(self) -> None:
        """A single observation must: set mean to the value, not detect drift,
        and return correct n_samples."""
        d = PageHinkleyDetector()
        result = d.update(0.75)
        assert not result
        assert d.n_samples == 1
        assert d.reference_mean == pytest.approx(0.75)

    def test_identical_values_never_drift(self) -> None:
        """Feeding the exact same value forever must never trigger drift.
        The cumsum deviation stays at zero because (value - mean) → 0 as mean → value."""
        d = PageHinkleyDetector(delta=0.001, lambda_=1.0)
        for _ in range(500):
            if d.update(0.42):
                pytest.fail(f"Drift detected on constant stream at step {d.n_samples}")
        assert not d.is_drift_detected

    def test_reset_preserves_mean(self) -> None:
        """reset() must clear cumsums but keep reference_mean exactly."""
        d = PageHinkleyDetector()
        for _ in range(100):
            d.update(0.7)
        mean_before = d.reference_mean
        d.reset()
        assert not d.is_drift_detected
        assert d.reference_mean == pytest.approx(mean_before)

    def test_full_reset_zeros_mean(self) -> None:
        """full_reset() must zero everything including the mean."""
        d = PageHinkleyDetector()
        for _ in range(100):
            d.update(0.7)
        d.full_reset()
        assert d.n_samples == 0
        assert d.reference_mean == 0.0
        assert not d.is_drift_detected

    def test_large_lambda_never_triggers(self) -> None:
        """lambda_=1e12 must never trigger drift on any realistic finite stream."""
        d = PageHinkleyDetector(delta=0.0, lambda_=1e12)
        rng = np.random.default_rng(3)
        for _ in range(5000):
            if d.update(float(rng.uniform(-100, 100))):
                pytest.fail(f"Drift detected with lambda_=1e12 at step {d.n_samples}")

    def test_delta_equals_zero(self) -> None:
        """delta=0.0 is the boundary valid value. Must not raise."""
        d = PageHinkleyDetector(delta=0.0)
        assert d.delta == 0.0

    def test_is_drift_detected_persists_until_reset(self) -> None:
        """After update() returns True, is_drift_detected must stay True until reset()."""
        d = PageHinkleyDetector(delta=0.0, lambda_=1.0)
        rng = np.random.default_rng(4)
        # Build mean at 0.5, then shift abruptly to 1.5
        for _ in range(200):
            d.update(float(rng.normal(0.5, 0.1)))
        detected_on_update = False
        for _ in range(500):
            if d.update(float(rng.normal(1.5, 0.1))):
                detected_on_update = True
                break
        assert detected_on_update
        assert d.is_drift_detected

        # Subsequent calls with same signal: is_drift_detected stays True
        for _ in range(10):
            d.update(float(rng.normal(1.5, 0.1)))
        assert d.is_drift_detected

        d.reset()
        assert not d.is_drift_detected
