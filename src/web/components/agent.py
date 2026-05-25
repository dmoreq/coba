"""Agent zone components — knowledge table, pull counter, policy state card."""

from __future__ import annotations

from typing import Any

from web.theme import FontScale, SpacingScale
from web.theme.theme_manager import ThemeManager

try:
    import flet as ft
except ModuleNotFoundError:  # pragma: no cover
    ft = None  # type: ignore[assignment]


def build_knowledge_table(
    page: Any,
    arm_labels: list[str],
    mean_rewards: list[float],
    pull_counts: list[int],
) -> Any:
    """Build a knowledge table showing estimated mean rewards per arm."""
    if ft is None:
        return None
    tokens = ThemeManager.get_tokens(page)

    if not arm_labels:
        return ft.Text(
            value="No arm data yet", size=FontScale.SMALL, color=tokens.text_muted, italic=True
        )

    rows: list[Any] = []
    for label, mean, pulls in zip(arm_labels, mean_rewards, pull_counts):
        mean_str = f"~{mean:.3f}" if mean > 0 else "—"
        rows.append(
            ft.Row(
                controls=[
                    ft.Text(
                        value=label, size=FontScale.SMALL, weight=ft.FontWeight.W_500, expand=1
                    ),
                    ft.Text(
                        value=mean_str, size=FontScale.SMALL, color=tokens.agent_accent, width=60
                    ),
                    ft.Text(
                        value=f"({pulls})",
                        size=FontScale.CAPTION,
                        color=tokens.text_muted,
                        width=40,
                    ),
                ],
                spacing=SpacingScale.XS,
                tight=True,
            )
        )

    return ft.Column(controls=rows, spacing=SpacingScale.XS, tight=True)


def build_pull_counter(page: Any, arm_labels: list[str], pull_counts: list[int]) -> Any:
    """Build a compact horizontal bar chart of pull counts."""
    if ft is None:
        return None
    tokens = ThemeManager.get_tokens(page)

    if not arm_labels or max(pull_counts) == 0:
        return ft.Text(
            value="No pulls yet", size=FontScale.SMALL, color=tokens.text_muted, italic=True
        )

    max_count = max(pull_counts)
    bar_rows: list[Any] = []
    for label, count in zip(arm_labels, pull_counts):
        width_pct = count / max_count if max_count > 0 else 0
        bar_rows.append(
            ft.Row(
                controls=[
                    ft.Text(
                        value=label, size=FontScale.CAPTION, color=tokens.text_secondary, width=80
                    ),
                    ft.Container(
                        width=int(100 * width_pct),
                        height=12,
                        bgcolor=tokens.agent_accent,
                        border_radius=2,
                    ),
                    ft.Text(
                        value=str(count), size=FontScale.CAPTION, color=tokens.text_muted, width=30
                    ),
                ],
                spacing=SpacingScale.XS,
                tight=True,
            )
        )

    return ft.Column(controls=bar_rows, spacing=2, tight=True)


def build_policy_state_card(
    page: Any, policy_id: str, policy_data: dict[str, Any] | None = None
) -> Any:
    """Build a policy state card showing algorithm parameters."""
    if ft is None:
        return None
    tokens = ThemeManager.get_tokens(page)

    if not policy_data:
        policy_data = {}

    rows: list[Any] = [
        ft.Row(
            controls=[
                ft.Text(
                    value="Policy:",
                    size=FontScale.SMALL,
                    weight=ft.FontWeight.W_500,
                    color=tokens.text_secondary,
                ),
                ft.Text(value=policy_id, size=FontScale.SMALL, color=tokens.agent_accent),
            ],
            spacing=SpacingScale.XS,
            tight=True,
        )
    ]

    for key, value in policy_data.items():
        rows.append(
            ft.Row(
                controls=[
                    ft.Text(
                        value=f"{key}:", size=FontScale.CAPTION, color=tokens.text_muted, width=60
                    ),
                    ft.Text(value=str(value), size=FontScale.CAPTION, color=tokens.text_primary),
                ],
                spacing=SpacingScale.XS,
                tight=True,
            )
        )

    return ft.Column(controls=rows, spacing=SpacingScale.XS, tight=True)
