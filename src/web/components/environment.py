"""Environment zone components — world card, context display, hidden truth panel."""

from __future__ import annotations

from typing import Any

from web.theme import FontScale, SpacingScale
from web.theme.theme_manager import ThemeManager

try:
    import flet as ft
except ModuleNotFoundError:  # pragma: no cover
    ft = None  # type: ignore[assignment]


def build_world_card(page: Any, title: str, description: str) -> Any:
    """Build the world info card with title and description."""
    if ft is None:
        return None
    tokens = ThemeManager.get_tokens(page)
    return ft.Column(
        controls=[
            ft.Text(
                value=title,
                size=FontScale.BODY,
                weight=ft.FontWeight.W_600,
                color=tokens.environment_accent,
            ),
            ft.Text(value=description, size=FontScale.SMALL, color=tokens.text_secondary),
        ],
        spacing=SpacingScale.XS,
        tight=True,
    )


def build_context_display(
    page: Any, context: dict[str, Any], feature_order: list[str] | None = None
) -> Any:
    """Build a context feature display with key-value pairs."""
    if ft is None:
        return None
    tokens = ThemeManager.get_tokens(page)
    controls: list[Any] = []

    keys = feature_order if feature_order else list(context.keys())
    for key in keys:
        if key == "step":
            continue
        value = context.get(key, "—")
        controls.append(
            ft.Row(
                controls=[
                    ft.Text(
                        value=f"{key}:",
                        size=FontScale.SMALL,
                        weight=ft.FontWeight.W_500,
                        color=tokens.text_secondary,
                    ),
                    ft.Text(value=str(value), size=FontScale.SMALL, color=tokens.text_primary),
                ],
                spacing=SpacingScale.XS,
                tight=True,
            )
        )

    if not controls:
        controls.append(
            ft.Text(
                value="No context features",
                size=FontScale.SMALL,
                color=tokens.text_muted,
                italic=True,
            )
        )

    return ft.Column(controls=controls, spacing=SpacingScale.XS, tight=True)
