"""
PageHinkleyDetector — sequential drift detection for reward streams.

The Page-Hinkley test monitors a stream of scalar values and flags a
distributional shift when the cumulative deviation from the reference mean
exceeds a threshold. It detects *persistent* shifts (not transient noise)
because the signal must accumulate over multiple observations.

Two-sided variant monitors both upward (reward improvement) and downward
(reward degradation) shifts simultaneously.

Integration with ClusterBandit:
  Instantiate one PageHinkleyDetector per arm. Feed each observed reward to
  the arm's detector after every update(). When drift is detected, reset
  the arm's models across all clusters so the bandit re-learns the new
  reward distribution from scratch.

  See ClusterBandit(enable_drift_detection=True) for automatic integration,
  or examples/13_drift_detection.py for manual usage.
"""

from __future__ import annotations

from loguru import logger


class PageHinkleyDetector:
    """Two-sided Page-Hinkley sequential drift detector.

    Tracks cumulative deviations from a running mean estimate. Signals drift
    when the accumulated deviation (in either direction) exceeds `lambda_`.

    Args:
        delta: Minimum magnitude of change to detect. Acts as a small bias
               subtracted from each step to suppress micro-fluctuations.
               Lower → more sensitive (more false alarms).
               Higher → only large shifts detected. Default 0.005.
        lambda_: Detection threshold. The cumulative sum must exceed this
                 value before drift is declared. Higher → fewer false alarms
                 but slower detection. Default 50.0.
        alpha: Exponential moving average weight for the reference mean
               (0 < alpha < 1). Values close to 1 give the mean more inertia
               (stable reference). Default 0.999.

    Example::

        detector = PageHinkleyDetector(delta=0.01, lambda_=30.0)
        for reward in reward_stream:
            if detector.update(reward):
                print("Drift detected — resetting arm model")
                arm_model.reset()
                detector.reset()  # keep mean, restart cumulative sums
    """

    def __init__(
        self,
        delta: float = 0.005,
        lambda_: float = 50.0,
        alpha: float = 0.999,
    ) -> None:
        if delta < 0:
            raise ValueError(f"delta must be >= 0, got {delta}")
        if lambda_ <= 0:
            raise ValueError(f"lambda_ must be > 0, got {lambda_}")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha

        self._n: int = 0
        self._mean: float = 0.0

        # Both sides use the same PH structure: cumsum - running_min > lambda_
        # cumsum_up   += x - mean - delta  → increases on upward shift
        # cumsum_down += mean - x - delta  → increases on downward shift
        self._cumsum_up: float = 0.0
        self._min_up: float = float("inf")

        self._cumsum_down: float = 0.0
        self._min_down: float = float("inf")

        self._drift_detected: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, value: float) -> bool:
        """Process a new observation.

        Args:
            value: New scalar reward (or any scalar signal to monitor).

        Returns:
            True if drift is detected, False otherwise.
        """
        self._n += 1

        # Update exponential moving mean (first sample bootstraps the mean)
        if self._n == 1:
            self._mean = value
        else:
            self._mean = self.alpha * self._mean + (1.0 - self.alpha) * value

        # Update two-sided cumulative sums
        self._cumsum_up += value - self._mean - self.delta
        self._cumsum_down += self._mean - value - self.delta

        # Track running minima — the gap (cumsum - min) grows after a shift
        if self._cumsum_up < self._min_up:
            self._min_up = self._cumsum_up
        if self._cumsum_down < self._min_down:
            self._min_down = self._cumsum_down

        # Drift when gap between current cumsum and its historical minimum exceeds threshold
        drift_up = (self._cumsum_up - self._min_up) > self.lambda_
        drift_down = (self._cumsum_down - self._min_down) > self.lambda_

        self._drift_detected = drift_up or drift_down

        if self._drift_detected:
            direction = "upward" if drift_up else "downward"
            logger.debug(
                "PageHinkley drift ({dir}) after {n} samples, mean={m:.4f}",
                dir=direction,
                n=self._n,
                m=self._mean,
            )

        return self._drift_detected

    def reset(self) -> None:
        """Reset cumulative sums but keep the current mean estimate.

        Call this after handling a detected drift. The mean estimate carries
        forward as the new reference baseline — only the cumulative deviation
        counters restart from zero.
        """
        self._cumsum_up = 0.0
        self._min_up = float("inf")
        self._cumsum_down = 0.0
        self._min_down = float("inf")
        self._drift_detected = False

    def full_reset(self) -> None:
        """Reset everything including the mean estimate.

        Use when the arm is completely removed and re-added, or when a fresh
        start with no prior reference is desired.
        """
        self._n = 0
        self._mean = 0.0
        self.reset()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_drift_detected(self) -> bool:
        """True if the last call to update() triggered a drift alarm."""
        return self._drift_detected

    @property
    def n_samples(self) -> int:
        """Total number of observations seen since last full_reset()."""
        return self._n

    @property
    def reference_mean(self) -> float:
        """Current exponential moving mean of the monitored signal."""
        return self._mean
