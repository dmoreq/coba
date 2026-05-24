"""Treatment card view-model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TreatmentCardModel:
    """View-model for an action card."""

    arm_id: str
    label: str
    predicted_score: float | None = None
    selected: bool = False
