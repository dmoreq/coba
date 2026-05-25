"""Semantic color tokens for light and dark themes.

Every component reads colors from ColorTokens — never hardcodes hex values.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColorTokens:
    """Semantic color tokens used by all UI components."""

    # Surfaces
    bg_primary: str
    bg_secondary: str
    bg_tertiary: str
    surface_border: str

    # Zones (semantic tints)
    environment_zone_bg: str
    agent_zone_bg: str
    interaction_zone_bg: str

    # Accents
    environment_accent: str
    agent_accent: str
    success_feedback: str
    regret_feedback: str
    selected_glow: str

    # Text
    text_primary: str
    text_secondary: str
    text_muted: str
    text_on_accent: str

    # Charts
    chart_bg: str
    chart_grid: str
    chart_line_primary: str
    chart_line_secondary: str
    chart_bar_fill: str

    # Controls
    control_bg: str
    control_border: str
    control_fg: str
    slider_track: str
    slider_thumb: str


LIGHT_TOKENS = ColorTokens(
    bg_primary="#FAFAFA",
    bg_secondary="#FFFFFF",
    bg_tertiary="#F5F5F5",
    surface_border="#E0E0E0",
    environment_zone_bg="#F0F7FA",
    agent_zone_bg="#FFF8F0",
    interaction_zone_bg="#FFFFFF",
    environment_accent="#00796B",
    agent_accent="#E65100",
    success_feedback="#2E7D32",
    regret_feedback="#C62828",
    selected_glow="#FFB74D",
    text_primary="#212121",
    text_secondary="#616161",
    text_muted="#9E9E9E",
    text_on_accent="#FFFFFF",
    chart_bg="#00000000",
    chart_grid="#E0E0E0",
    chart_line_primary="#00796B",
    chart_line_secondary="#E65100",
    chart_bar_fill="#90A4AE",
    control_bg="#FFFFFF",
    control_border="#BDBDBD",
    control_fg="#212121",
    slider_track="#BDBDBD",
    slider_thumb="#00796B",
)

DARK_TOKENS = ColorTokens(
    bg_primary="#121212",
    bg_secondary="#1E1E1E",
    bg_tertiary="#2C2C2C",
    surface_border="#333333",
    environment_zone_bg="#0D2028",
    agent_zone_bg="#281A0A",
    interaction_zone_bg="#1E1E1E",
    environment_accent="#4DB6AC",
    agent_accent="#FFB74D",
    success_feedback="#66BB6A",
    regret_feedback="#EF5350",
    selected_glow="#FF8F00",
    text_primary="#E0E0E0",
    text_secondary="#9E9E9E",
    text_muted="#616161",
    text_on_accent="#121212",
    chart_bg="#00000000",
    chart_grid="#333333",
    chart_line_primary="#4DB6AC",
    chart_line_secondary="#FFB74D",
    chart_bar_fill="#546E7A",
    control_bg="#2C2C2C",
    control_border="#444444",
    control_fg="#E0E0E0",
    slider_track="#444444",
    slider_thumb="#4DB6AC",
)
