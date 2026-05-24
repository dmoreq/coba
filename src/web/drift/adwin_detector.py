"""ADWIN-based drift detector (sliding window approach)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ADWINDriftDetector:
    """Adaptive sliding window drift detector.

    Maintains a window of recent observations. When the mean of two
    sub-windows diverges beyond a confidence threshold, a drift is
    triggered and old observations are dropped.
    """

    delta: float = 0.002
    min_window_length: int = 10
    max_window_length: int = 500

    _window: list[float] = field(default_factory=list, init=False)
    _events: list[tuple[int, float, float]] = field(default_factory=list, init=False)

    def reset(self) -> None:
        self._window.clear()
        self._events.clear()

    def update(self, value: float, step_index: int = 0) -> bool:
        self._window.append(value)
        if len(self._window) > self.max_window_length:
            self._window.pop(0)

        if len(self._window) < 2 * self.min_window_length:
            return False

        n = len(self._window)
        for split in range(self.min_window_length, n - self.min_window_length + 1):
            left = self._window[:split]
            right = self._window[split:]

            mean_left = sum(left) / float(len(left))
            mean_right = sum(right) / float(len(right))

            var_left = max(1e-9, sum((x - mean_left) ** 2 for x in left) / float(len(left)))
            var_right = max(1e-9, sum((x - mean_right) ** 2 for x in right) / float(len(right)))

            pooled_std = math.sqrt(var_left / float(len(left)) + var_right / float(len(right)))
            if pooled_std < 1e-9:
                continue

            epsilon_cut = math.sqrt(
                2.0 / float(len(left)) * var_left * math.log(2.0 / self.delta)
                + 2.0 / (3.0 * float(len(left))) * 3.0 * math.log(2.0 / self.delta)
            ) + math.sqrt(
                2.0 / float(len(right)) * var_right * math.log(2.0 / self.delta)
                + 2.0 / (3.0 * float(len(right))) * 3.0 * math.log(2.0 / self.delta)
            )

            if abs(mean_left - mean_right) > epsilon_cut:
                self._events.append((step_index, value, mean_left))
                self._window = self._window[split:]
                return True

        return False

    @property
    def reference_mean(self) -> float:
        if not self._window:
            return 0.0
        return sum(self._window) / float(len(self._window))

    @property
    def events(self) -> list[tuple[int, float, float]]:
        return list(self._events)
