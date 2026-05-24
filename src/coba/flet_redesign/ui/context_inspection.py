"""Context inspection model for contextual lessons."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextInspectionModel:
    """Feature vector and per-feature contribution preview."""

    feature_order: tuple[str, ...]
    feature_values: tuple[float, ...]
    notes: str
