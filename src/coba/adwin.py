"""
ADWIN (ADaptive WINdowing) drift detector.

Reference:
  Bifet & Gavalda, "Learning from Time-Changing Data with Adaptive Windowing",
  SIAM 2007.

ADWIN maintains a variable-length window of recent observations and detects
drift when the means of two sub-windows differ by more than a statistically
significant threshold. The window shrinks automatically when drift is detected,
keeping only the more recent data.

Compared to PageHinkley:
  - PageHinkley is faster (O(1) per update) but assumes a persistent monotone
    shift and can miss oscillating or multi-modal drift.
  - ADWIN is more robust to different drift shapes (abrupt, gradual, recurring)
    at the cost of O(log n) amortized time and O(n) memory.

Use ADWIN when:
  - You don't know the drift direction.
  - Drift may be gradual or oscillating.
  - You want the window to adapt automatically without manual reset.

Use PageHinkley when:
  - You need O(1) updates in a high-throughput system.
  - Drift is expected to be persistent and unidirectional.

Integration with ClusterBandit:

    from coba.adwin import ADWINDetector

    detector = ADWINDetector(delta=0.002)
    for reward in reward_stream:
        if detector.update(reward):
            bandit.reset_arm(arm)
            detector.reset()
"""

from __future__ import annotations

import math
from collections import deque

from loguru import logger


class ADWINDetector:
    """Adaptive Windowing (ADWIN) drift detector.

    Maintains a bucket-based compression of the observation window and tests
    all possible splits for a statistically significant mean difference.

    Args:
        delta: Confidence parameter. Lower → more sensitive (more false alarms).
               Typical values: 0.001 – 0.01. Default 0.002.
        max_buckets: Controls compression granularity. Higher → more precise
                     but slower. Default 5 (matches reference implementation).

    Example::

        detector = ADWINDetector(delta=0.002)
        for reward in stream:
            if detector.update(reward):
                print(f"Drift! window reset to {detector.n_samples} obs")
    """

    def __init__(self, delta: float = 0.002, max_buckets: int = 5) -> None:
        if not (0.0 < delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if max_buckets < 1:
            raise ValueError(f"max_buckets must be >= 1, got {max_buckets}")

        self.delta = delta
        self.max_buckets = max_buckets

        # Each bucket stores (count, variance_sum, mean) for a run of observations.
        # We use a deque of lists for O(1) appends and pops from either end.
        self._buckets: deque[list[float]] = deque()  # each entry: [count, var_sum, mean]

        self._total_count: int = 0
        self._total_sum: float = 0.0
        self._total_var: float = 0.0

        self._drift_detected: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, value: float) -> bool:
        """Add a new observation and test for drift.

        Args:
            value: New scalar observation.

        Returns:
            True if drift is detected (window was shrunk), False otherwise.
        """
        # Insert new observation as a bucket of size 1
        self._insert(value)
        # Compress: merge adjacent equal-size buckets to bound memory
        self._compress_buckets()
        # Test for drift: find any split where the two sub-window means diverge
        detected = self._detect_change()
        self._drift_detected = detected
        return detected

    def reset(self) -> None:
        """Clear window and statistics (full reset, not just cumulative sums)."""
        self._buckets.clear()
        self._total_count = 0
        self._total_sum = 0.0
        self._total_var = 0.0
        self._drift_detected = False

    @property
    def is_drift_detected(self) -> bool:
        """True if the last call to update() triggered a drift alarm."""
        return self._drift_detected

    @property
    def n_samples(self) -> int:
        """Number of observations currently in the active window."""
        return self._total_count

    @property
    def window_mean(self) -> float:
        """Current mean of all observations in the active window."""
        if self._total_count == 0:
            return 0.0
        return self._total_sum / self._total_count

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _insert(self, value: float) -> None:
        """Append a new bucket of size 1."""
        self._buckets.appendleft([1, 0.0, value])
        self._total_count += 1
        self._total_sum += value
        self._total_var += 0.0  # variance of a single point is 0

    def _compress_buckets(self) -> None:
        """Merge adjacent buckets of equal size to keep memory O(log n)."""
        bucket_list = list(self._buckets)
        # Group by size; when more than max_buckets have the same size, merge two
        i = 0
        while i < len(bucket_list) - 1:
            b0 = bucket_list[i]
            b1 = bucket_list[i + 1]
            if b0[0] == b1[0]:
                # Count how many buckets of this size exist
                same_count = sum(1 for b in bucket_list if b[0] == b0[0])
                if same_count > self.max_buckets:
                    # Merge b0 and b1 into a bucket of size 2*b0[0]
                    n0, v0, m0 = b0
                    n1, v1, m1 = b1
                    n_merged = n0 + n1
                    mean_merged = (n0 * m0 + n1 * m1) / n_merged
                    # Variance by parallel formula
                    var_merged = v0 + v1 + (m0 - m1) ** 2 * n0 * n1 / n_merged
                    bucket_list[i] = [n_merged, var_merged, mean_merged]
                    bucket_list.pop(i + 1)
                    continue
            i += 1
        self._buckets = deque(bucket_list)

    def _detect_change(self) -> bool:
        """Slide a split point through the window; return True if drift found."""
        if self._total_count < 2:
            return False

        # Accumulated stats from the right (oldest) end
        n0, s0 = 0, 0.0

        bucket_list = list(self._buckets)
        # Iterate from oldest bucket toward newest
        for bucket in reversed(bucket_list):
            bn, _, bm = bucket
            n0 += bn
            s0 += bn * bm
            n1 = self._total_count - n0
            if n1 == 0:
                continue
            s1 = self._total_sum - s0
            m0 = s0 / n0
            m1 = s1 / n1

            # ADWIN threshold: epsilon_cut
            # |m0 - m1| > epsilon_cut  → drift
            epsilon_cut = math.sqrt(
                (1.0 / (2.0 * n0) + 1.0 / (2.0 * n1))
                * math.log(4.0 * self._total_count / self.delta)
            )

            if abs(m0 - m1) >= epsilon_cut:
                direction = "upward" if m1 > m0 else "downward"
                logger.debug(
                    "ADWIN drift ({dir}): |m0={m0:.4f} - m1={m1:.4f}| >= eps={eps:.4f}, " "n={n}",
                    dir=direction,
                    m0=m0,
                    m1=m1,
                    eps=epsilon_cut,
                    n=self._total_count,
                )
                # Shrink window: drop the oldest n0 observations
                self._shrink_window(n0)
                return True
        return False

    def _shrink_window(self, drop_count: int) -> None:
        """Remove the oldest ``drop_count`` observations from the window."""
        remaining = drop_count
        bucket_list = list(self._buckets)
        # Buckets are stored newest-first; oldest are at the end
        while remaining > 0 and bucket_list:
            bn, bv, bm = bucket_list[-1]
            if bn <= remaining:
                # Drop the entire bucket
                self._total_count -= bn
                self._total_sum -= bn * bm
                self._total_var -= bv
                remaining -= bn
                bucket_list.pop()
            else:
                # Partially drop from the oldest bucket
                new_n = bn - remaining
                # Can't precisely recover variance of sub-bucket; reset to 0
                bucket_list[-1] = [new_n, 0.0, bm]
                self._total_count -= remaining
                self._total_sum -= remaining * bm
                remaining = 0
        self._buckets = deque(bucket_list)
