"""Tooltip models for parameter guidance."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParamTooltip:
    """Pedagogical tooltip payload for one control."""

    title: str
    intuition: str
    formula: str
    tuning_hint: str
