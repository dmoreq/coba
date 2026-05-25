"""Interaction zone components — arm cards, reward feedback, step controls, loop visualizer."""

from __future__ import annotations

from typing import Any

from web.statemgmt.interaction_state import InteractionPhase
from web.theme import FontScale, SpacingScale
from web.theme.theme_manager import ThemeManager

try:
    import flet as ft
except ModuleNotFoundError:  # pragma: no cover
    ft = None  # type: ignore[assignment]


def build_arm_cards(
    page: Any,
    arm_labels: list[str],
    selected_arm: str | None = None,
    predicted_scores: list[float] | None = None,
) -> Any:
    """Build arm selection cards. The selected arm gets an amber glow."""
    if ft is None:
        return None
    tokens = ThemeManager.get_tokens(page)
    cards: list[Any] = []

    for i, label in enumerate(arm_labels):
        is_selected = label == selected_arm
        score = predicted_scores[i] if predicted_scores and i < len(predicted_scores) else None
        score_str = f" ({score:.3f})" if score is not None else ""

        cards.append(
            ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Text(
                            value=f"{label}{score_str}",
                            size=FontScale.SMALL,
                            weight=ft.FontWeight.W_500 if is_selected else None,
                            color=tokens.text_on_accent if is_selected else tokens.text_primary,
                        ),
                    ],
                    tight=True,
                ),
                padding=SpacingScale.SM,
                border_radius=6,
                bgcolor=tokens.selected_glow if is_selected else tokens.bg_tertiary,
                border=ft.border.all(2, tokens.agent_accent if is_selected else "transparent"),
                width=160,
            )
        )

    return ft.Row(
        controls=(
            cards if cards else [ft.Text("No arms available", size=FontScale.SMALL, italic=True)]
        ),
        spacing=SpacingScale.SM,
        wrap=True,
    )


def build_reward_feedback(page: Any, reward: float | None = None) -> Any:
    """Build reward feedback indicator — green for success, red for failure."""
    if ft is None:
        return None
    tokens = ThemeManager.get_tokens(page)

    if reward is None:
        return ft.Container(height=40)  # placeholder

    is_success = reward > 0
    color = tokens.success_feedback if is_success else tokens.regret_feedback
    icon = ft.Icons.CHECK_CIRCLE if is_success else ft.Icons.CANCEL
    text = "Success! ✓" if is_success else "No reward ✗"

    return ft.Container(
        content=ft.Row(
            controls=[
                ft.Icon(name=icon, color=color, size=20),
                ft.Text(value=text, size=FontScale.BODY, color=color, weight=ft.FontWeight.W_600),
            ],
            spacing=SpacingScale.XS,
            tight=True,
        ),
        padding=SpacingScale.SM,
    )


def build_loop_visualizer(page: Any, phase: InteractionPhase = InteractionPhase.IDLE) -> Any:
    """Build a visual indicator showing which of the 4 interaction phases is active."""
    if ft is None:
        return None
    tokens = ThemeManager.get_tokens(page)

    phases = [
        ("①", "Context", InteractionPhase.CONTEXT_GENERATED, tokens.environment_accent),
        ("②", "Arm", InteractionPhase.ARM_SELECTED, tokens.agent_accent),
        ("③", "Reward", InteractionPhase.REWARD_RECEIVED, tokens.success_feedback),
        ("④", "Learn", InteractionPhase.KNOWLEDGE_UPDATED, tokens.agent_accent),
    ]

    dots: list[Any] = []
    for num, label, ph, color in phases:
        is_active = phase == ph
        is_completed = phase is not InteractionPhase.IDLE and ph.value < phase.value
        dot_color = (
            color if is_active else (tokens.text_secondary if is_completed else tokens.text_muted)
        )

        dots.append(
            ft.Column(
                controls=[
                    ft.Container(
                        content=ft.Text(
                            num,
                            size=FontScale.TITLE,
                            weight=ft.FontWeight.BOLD,
                            color=tokens.text_on_accent if is_active else tokens.text_muted,
                        ),
                        width=36,
                        height=36,
                        border_radius=18,
                        bgcolor=dot_color if is_active or is_completed else tokens.bg_tertiary,
                        alignment=ft.alignment.center,
                    ),
                    ft.Text(
                        label,
                        size=FontScale.CAPTION,
                        color=dot_color if is_active else tokens.text_muted,
                    ),
                ],
                spacing=SpacingScale.XS,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                tight=True,
            )
        )
        if num != "④":
            dots.append(
                ft.Container(
                    content=ft.Icon(ft.Icons.ARROW_FORWARD, size=16, color=tokens.text_muted),
                )
            )

    return ft.Row(controls=dots, spacing=SpacingScale.XS, tight=True)


def build_step_controls(
    page: Any,
    current_step: int = 0,
    is_running: bool = False,
    *,
    on_step: Any = None,
    on_play: Any = None,
    on_reset: Any = None,
    on_run_n: Any = None,
) -> Any:
    """Build Step, Play/Pause, Reset, and Run-N controls."""
    if ft is None:
        return None

    step_label = f"  Step ({current_step})  "
    play_label = "  ⏸ Pause  " if is_running else "  ▶ Play  "

    return ft.Column(
        controls=[
            ft.Row(
                controls=[
                    ft.FilledTonalButton(
                        text=step_label, on_click=on_step, icon=ft.Icons.PLAY_ARROW
                    ),
                    ft.FilledTonalButton(text=play_label, on_click=on_play),
                    ft.FilledTonalButton(
                        text="  ↺ Reset  ", on_click=on_reset, icon=ft.Icons.RESTART_ALT
                    ),
                ],
                spacing=SpacingScale.SM,
                wrap=True,
            ),
        ],
        spacing=SpacingScale.SM,
        tight=True,
    )
