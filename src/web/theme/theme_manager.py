"""Theme manager — applies color tokens to Flet page and handles dark/light toggling.

Components access tokens via ThemeManager.get_tokens(page).
"""

from __future__ import annotations

from typing import Any

from web.theme.tokens import ColorTokens, DARK_TOKENS, LIGHT_TOKENS

try:
    import flet as ft
except ModuleNotFoundError:  # pragma: no cover
    ft = None  # type: ignore[assignment]


class ThemeManager:
    """Central theme manager for the COBA web app.

    Call apply_theme at startup and toggle when the user toggles dark/light mode.
    """

    TOKENS_KEY = "__coba_color_tokens"

    @staticmethod
    def apply_theme(page: Any, mode: str = "light") -> None:
        """Apply color tokens to the page and store them in session."""
        tokens = LIGHT_TOKENS if mode == "light" else DARK_TOKENS
        page.theme_mode = ft.ThemeMode.LIGHT if mode == "light" else ft.ThemeMode.DARK

        page.theme = ft.Theme(
            color_scheme_seed=tokens.environment_accent,
            font_family="system-ui, -apple-system, sans-serif",
        )
        page.dark_theme = ft.Theme(
            color_scheme_seed=DARK_TOKENS.environment_accent,
            font_family="system-ui, -apple-system, sans-serif",
        )

        page.session.set(ThemeManager.TOKENS_KEY, tokens)

    @staticmethod
    def get_tokens(page: Any) -> ColorTokens:
        """Retrieve current color tokens from page session."""
        tokens = page.session.get(ThemeManager.TOKENS_KEY)
        if tokens is None:
            return LIGHT_TOKENS
        return tokens

    @staticmethod
    def toggle(page: Any) -> ColorTokens:
        """Toggle between light and dark mode. Returns the new tokens."""
        current = ThemeManager.get_tokens(page)
        mode = "dark" if current == LIGHT_TOKENS else "light"
        ThemeManager.apply_theme(page, mode)
        return ThemeManager.get_tokens(page)
