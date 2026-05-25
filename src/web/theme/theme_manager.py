"""Theme manager — applies color tokens to Flet page and handles dark/light toggling.

Components access tokens via ThemeManager.get_tokens(page).

Stores current tokens on page.data (arbitrary user data field available on
every Flet control). Flet 0.85.1 does not expose page.session.set/get.
"""

from __future__ import annotations

from typing import Any

from web.theme.tokens import ColorTokens, DARK_TOKENS, LIGHT_TOKENS

try:
    import flet as ft
except ModuleNotFoundError:  # pragma: no cover
    ft = None  # type: ignore[assignment]

_TOKENS_ATTR = "__coba_color_tokens"


class ThemeManager:
    """Central theme manager for the COBA web app.

    Call apply_theme at startup and toggle when the user toggles dark/light mode.
    """

    @staticmethod
    def apply_theme(page: Any, mode: str = "light") -> None:
        """Apply color tokens to the page and store them on page.data."""
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

        # Store tokens on page.data (generic user-data attribute)
        page.data = {_TOKENS_ATTR: tokens}

    @staticmethod
    def get_tokens(page: Any) -> ColorTokens:
        """Retrieve current color tokens from page.data."""
        if isinstance(getattr(page, "data", None), dict):
            tokens = page.data.get(_TOKENS_ATTR)
            if tokens is not None:
                return tokens
        return LIGHT_TOKENS

    @staticmethod
    def toggle(page: Any) -> ColorTokens:
        """Toggle between light and dark mode. Returns the new tokens."""
        current = ThemeManager.get_tokens(page)
        mode = "dark" if current == LIGHT_TOKENS else "light"
        ThemeManager.apply_theme(page, mode)
        return ThemeManager.get_tokens(page)
