"""Theme constants — spacing, typography, and animation durations."""

from __future__ import annotations


class SpacingScale:
    """Consistent spacing scale used across all components."""

    XS: int = 4
    SM: int = 8
    MD: int = 12
    LG: int = 20
    XL: int = 32


class FontScale:
    """Consistent font size scale used across all components."""

    CAPTION: int = 10
    SMALL: int = 12
    BODY: int = 14
    TITLE: int = 18
    HEADING: int = 24


class AnimationDurations:
    """Animation durations in milliseconds."""

    PHASE_CONTEXT: int = 300
    PHASE_ARM: int = 400
    PHASE_REWARD: int = 600
    PHASE_KNOWLEDGE: int = 300
    CHART: int = 300
    FEEDBACK_FADE: int = 600
