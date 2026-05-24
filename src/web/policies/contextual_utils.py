"""Shared helpers for contextual policy feature encoding."""

from __future__ import annotations

from typing import Any


def context_to_vector(context: dict[str, Any], feature_order: tuple[str, ...]) -> list[float]:
    """Convert mixed-type context payload into a numeric feature vector."""
    vector: list[float] = []
    for key in feature_order:
        value = context.get(key, 0.0)
        vector.append(_to_float(value))
    return vector


def _to_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(sum(ord(ch) for ch in value) % 97) / 96.0
    return 0.0
