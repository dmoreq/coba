"""Theme toggle component — dark/light mode switch with sun/moon icon."""

from __future__ import annotations

from typing import Any

from web.session import get_shell
from web.theme.theme_manager import ThemeManager

try:
    import flet as ft
except ModuleNotFoundError:  # pragma: no cover
    ft = None  # type: ignore[assignment]


def build_theme_toggle(page: Any) -> Any:
    """Build a dark/light mode toggle icon button."""
    if ft is None:
        return None

    is_dark = page.theme_mode == ft.ThemeMode.DARK

    def _on_toggle(e: Any) -> None:
        ThemeManager.toggle(page)
        page.update()
        shell = get_shell(page)
        if shell:
            shell._refresh_view()

    return ft.IconButton(
        icon=ft.Icons.DARK_MODE if is_dark else ft.Icons.LIGHT_MODE,
        tooltip="Toggle dark mode",
        on_click=_on_toggle,
        icon_size=20,
    )
