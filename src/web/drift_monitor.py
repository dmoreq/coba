"""Drift detection timeline utilities."""

from __future__ import annotations

from dataclasses import dataclass

from coba.drift import PageHinkleyDetector


@dataclass(frozen=True)
class DriftEvent:
    """One drift event emitted by the monitor."""

    step_index: int
    value: float
    reference_mean: float


class DriftTimeline:
    """Wrap Page-Hinkley detector with explicit event timeline state."""

    def __init__(self, delta: float = 0.005, lambda_: float = 20.0, alpha: float = 0.999) -> None:
        self.detector = PageHinkleyDetector(delta=delta, lambda_=lambda_, alpha=alpha)
        self.events: list[DriftEvent] = []

    def update(self, *, step_index: int, value: float) -> bool:
        detected = self.detector.update(value)
        if detected:
            self.events.append(
                DriftEvent(
                    step_index=step_index,
                    value=value,
                    reference_mean=self.detector.reference_mean,
                )
            )
            self.detector.reset()
        return detected

    def indicators(self) -> dict[str, float | int]:
        latest_step = self.events[-1].step_index if self.events else -1
        return {
            "num_events": len(self.events),
            "latest_step": latest_step,
            "reference_mean": self.detector.reference_mean,
        }
