"""Tests for theme tokens and theme manager."""

from __future__ import annotations


def test_light_and_dark_tokens_have_same_fields() -> None:
    from web.theme.tokens import ColorTokens

    light_fields = set(f.name for f in ColorTokens.__dataclass_fields__.values())  # type: ignore[attr-defined]
    dark_fields = set(f.name for f in ColorTokens.__dataclass_fields__.values())  # type: ignore[attr-defined]
    assert light_fields == dark_fields


def test_tokens_are_frozen() -> None:
    from web.theme.tokens import LIGHT_TOKENS

    import dataclasses

    assert dataclasses.is_dataclass(LIGHT_TOKENS)


def test_text_contrast_on_primary_surface() -> None:
    """Verify text_primary vs bg_primary has sufficient contrast ratio."""
    from web.theme.tokens import DARK_TOKENS, LIGHT_TOKENS

    def _relative_luminance(hex_color: str) -> float:
        hex_color = hex_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        rs = r / 255.0
        gs = g / 255.0
        bs = b / 255.0

        def linearize(c: float) -> float:
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

        return 0.2126 * linearize(rs) + 0.7152 * linearize(gs) + 0.0722 * linearize(bs)

    for name, tokens in [("light", LIGHT_TOKENS), ("dark", DARK_TOKENS)]:
        l1 = _relative_luminance(tokens.text_primary)
        l2 = _relative_luminance(tokens.bg_primary)
        lighter = max(l1, l2)
        darker = min(l1, l2)
        ratio = (lighter + 0.05) / (darker + 0.05)
        assert ratio >= 4.5, f"{name} theme text/bg contrast {ratio:.2f} < 4.5"


def test_spacing_constants_positive() -> None:
    from web.theme.constants import SpacingScale

    assert SpacingScale.XS > 0
    assert SpacingScale.SM > SpacingScale.XS
    assert SpacingScale.MD > SpacingScale.SM
    assert SpacingScale.LG > SpacingScale.MD
    assert SpacingScale.XL > SpacingScale.LG


def test_font_scale_ordered() -> None:
    from web.theme.constants import FontScale

    assert (
        FontScale.CAPTION < FontScale.SMALL < FontScale.BODY < FontScale.TITLE < FontScale.HEADING
    )


def test_animation_durations_positive() -> None:
    from web.theme.constants import AnimationDurations

    assert AnimationDurations.PHASE_CONTEXT > 0
    assert AnimationDurations.PHASE_ARM > 0
    assert AnimationDurations.PHASE_REWARD > 0
    assert AnimationDurations.PHASE_KNOWLEDGE > 0
    assert AnimationDurations.CHART > 0
