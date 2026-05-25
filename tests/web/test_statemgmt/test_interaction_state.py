"""Tests for interaction phase enum."""

from __future__ import annotations

from web.statemgmt.interaction_state import InteractionPhase


def test_phases_in_correct_order() -> None:
    phases = list(InteractionPhase)
    expected = ["IDLE", "CONTEXT_GENERATED", "ARM_SELECTED", "REWARD_RECEIVED", "KNOWLEDGE_UPDATED"]
    assert [p.name for p in phases] == expected
