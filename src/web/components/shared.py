"""Shared UI components — section headers, metric badges, empty states."""

from __future__ import annotations

from typing import Any

from web.theme import FontScale, SpacingScale
from web.theme.theme_manager import ThemeManager

try:
    import flet as ft
except ModuleNotFoundError:  # pragma: no cover
    ft = None  # type: ignore[assignment]


def build_section_header(page: Any, title: str, accent_color: str) -> Any:
    """Build a zone header with accent stripe below the title."""
    if ft is None:
        return None
    return ft.Column(
        controls=[
            ft.Text(value=title, size=FontScale.SMALL, weight=ft.FontWeight.BOLD),
            ft.Container(height=2, bgcolor=accent_color, border_radius=1),
        ],
        spacing=SpacingScale.XS,
        tight=True,
    )


def build_metric_badge(page: Any, label: str, value: str) -> Any:
    """Build a compact metric badge (value + label)."""
    if ft is None:
        return None
    tokens = ThemeManager.get_tokens(page)
    return ft.Column(
        controls=[
            ft.Text(value=value, size=FontScale.BODY, weight=ft.FontWeight.BOLD),
            ft.Text(value=label, size=FontScale.CAPTION, color=tokens.text_muted),
        ],
        spacing=SpacingScale.XS,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        tight=True,
    )


def build_empty_state(page: Any, message: str = "Nothing to show yet.") -> Any:
    """Build a centered empty state placeholder."""
    if ft is None:
        return None
    tokens = ThemeManager.get_tokens(page)
    return ft.Container(
        content=ft.Column(
            controls=[
                ft.Icon(name=ft.Icons.INFO_OUTLINE, size=32, color=tokens.text_muted),
                ft.Text(value=message, size=FontScale.SMALL, color=tokens.text_muted),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=SpacingScale.SM,
        ),
        padding=SpacingScale.XL,
        alignment=ft.alignment.center,
    )
