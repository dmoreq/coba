"""Contextual world presets for lesson scenarios."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextualPreset:
    """Preset metadata for contextual lesson scenarios."""

    preset_id: str
    world_id: str
    title: str
    description: str
    horizon: int


CONTEXTUAL_PRESETS: tuple[ContextualPreset, ...] = (
    ContextualPreset(
        preset_id="clinic_shift_morning",
        world_id="rural_clinic",
        title="Clinic Shift Morning",
        description="Higher comorbidity load with moderate symptom severity variance.",
        horizon=200,
    ),
    ContextualPreset(
        preset_id="moviematch_new_users",
        world_id="moviematch",
        title="New User Surge",
        description="Session history is sparse, requiring stronger feature exploration.",
        horizon=220,
    ),
    ContextualPreset(
        preset_id="newsfeed_breaking_mix",
        world_id="newsfeed",
        title="Breaking News Spike",
        description="Topic preference shifts quickly while attention depth fluctuates.",
        horizon=220,
    ),
)


def list_contextual_presets() -> tuple[ContextualPreset, ...]:
    return CONTEXTUAL_PRESETS
