"""Interaction phase enum and animation state."""

from __future__ import annotations

from enum import Enum, auto


class InteractionPhase(Enum):
    """The four phases of the interaction loop animation."""

    IDLE = auto()
    CONTEXT_GENERATED = auto()
    ARM_SELECTED = auto()
    REWARD_RECEIVED = auto()
    KNOWLEDGE_UPDATED = auto()
