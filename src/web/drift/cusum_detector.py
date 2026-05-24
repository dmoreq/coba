"""CUSUM-based drift detector."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CUSUMDriftDetector:
    """Cumulative sum drift detector with configurable threshold.

    Accumulates deviations from a reference mean. When the cumulative sum
    exceeds the threshold, a drift event is triggered and the detector resets.
    """

    threshold: float = 2.0
    drift_magnitude: float = 0.5
    warmup_steps: int = 20

    _cumulative_positive: float = field(default=0.0, init=False)
    _cumulative_negative: float = field(default=0.0, init=False)
    _reference_mean: float = field(default=0.0, init=False)
    _n_obs: int = field(default=0, init=False)
    _sum: float = field(default=0.0, init=False)
    _events: list[tuple[int, float, float]] = field(default_factory=list, init=False)

    def reset(self) -> None:
        self._cumulative_positive = 0.0
        self._cumulative_negative = 0.0
        self._reference_mean = 0.0
        self._n_obs = 0
        self._sum = 0.0
        self._events.clear()

    def update(self, value: float, step_index: int = 0) -> bool:
        self._n_obs += 1
        self._sum += value
        self._reference_mean = self._sum / float(self._n_obs)

        if self._n_obs < self.warmup_steps:
            return False

        deviation = value - self._reference_mean
        self._cumulative_positive = max(
            0.0, self._cumulative_positive + deviation - self.drift_magnitude
        )
        self._cumulative_negative = max(
            0.0, self._cumulative_negative - deviation - self.drift_magnitude
        )

        if self._cumulative_positive > self.threshold or self._cumulative_negative > self.threshold:
            self._events.append((step_index, value, self._reference_mean))
            self._cumulative_positive = 0.0
            self._cumulative_negative = 0.0
            return True
        return False

    @property
    def reference_mean(self) -> float:
        return self._reference_mean

    @property
    def events(self) -> list[tuple[int, float, float]]:
        return list(self._events)
