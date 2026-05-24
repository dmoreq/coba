"""Flet app entrypoint for the redesign shell."""

from __future__ import annotations

from typing import Any

from coba.flet_redesign.ui.preferences import PreferencesStore, UserPreferences
from coba.flet_redesign.ui.view_models import RouteUIModel, build_route_ui_model

try:
    import flet as ft
except ModuleNotFoundError:  # pragma: no cover - exercised by tests via run()
    ft = None  # type: ignore[assignment]


def _render_shell_view(view: RouteUIModel) -> Any:
    """Render one shell view using Flet controls."""
    if ft is None:
        raise RuntimeError("Flet is not installed. Install it with `pip install flet`.")

    content_controls: list[Any] = [
        ft.Text(value=view.heading, size=28, weight=ft.FontWeight.BOLD),
        ft.Text(value=view.description, size=14),
    ]
    if view.layout and view.scene_panel is not None:
        left_panel = ft.Container(
            expand=view.layout.left.width_ratio,
            padding=12,
            border=ft.border.all(1, "#D5D7DA"),
            border_radius=8,
            content=ft.Column(
                controls=[
                    ft.Text(value=view.layout.left.title, weight=ft.FontWeight.BOLD),
                    ft.Text(value=view.scene_panel.world_title),
                    ft.Text(value=view.scene_panel.world_description, size=12),
                ]
                + [
                    ft.Text(value=f"{key}: {value}", size=11)
                    for key, value in view.scene_panel.context_items.items()
                    if key != "step"
                ],
                tight=True,
                spacing=6,
            ),
        )
        center_panel = ft.Container(
            expand=view.layout.center.width_ratio,
            padding=12,
            border=ft.border.all(1, "#D5D7DA"),
            border_radius=8,
            content=ft.Column(
                controls=[
                    ft.Text(value=view.layout.center.title, weight=ft.FontWeight.BOLD),
                    ft.Row(
                        controls=[
                            ft.ElevatedButton(text=label) for label in view.run_control_labels
                        ],
                        wrap=True,
                    ),
                ]
                + [ft.ElevatedButton(text=card.label) for card in view.treatment_cards],
                spacing=8,
            ),
        )
        right_panel = ft.Container(
            expand=view.layout.right.width_ratio,
            padding=12,
            border=ft.border.all(1, "#D5D7DA"),
            border_radius=8,
            content=ft.Column(
                controls=[ft.Text(value=view.layout.right.title, weight=ft.FontWeight.BOLD)]
                + (
                    [
                        ft.Text(
                            value=f"Trace rows: {len(view.trace_records)}",
                            size=12,
                        ),
                        ft.Text(
                            value=f"Reward points: {len(view.arena_metrics.reward_series)}",
                            size=12,
                        ),
                        ft.Text(
                            value=f"Regret points: {len(view.arena_metrics.regret_series)}",
                            size=12,
                        ),
                    ]
                    if view.arena_metrics is not None
                    else []
                )
                + (
                    [
                        ft.Text(
                            value=f"Lesson: {view.lesson_panel.lesson_title}",
                            size=12,
                            weight=ft.FontWeight.BOLD,
                        ),
                        ft.Text(value=f"Stage: {view.lesson_panel.stage_index}", size=12),
                        ft.Text(value=view.lesson_panel.objective_text, size=12),
                        ft.Text(value=view.lesson_panel.step_explanation, size=11),
                    ]
                    if view.lesson_panel is not None
                    else []
                )
                + (
                    [
                        ft.Text(value="Context Vector", size=12, weight=ft.FontWeight.BOLD),
                        ft.Text(
                            value=", ".join(
                                f"{name}={value:.3f}"
                                for name, value in zip(
                                    view.context_inspection.feature_order,
                                    view.context_inspection.feature_values,
                                )
                            ),
                            size=11,
                        ),
                    ]
                    if view.context_inspection is not None
                    else []
                )
                + [
                    ft.Text(
                        value=f"Debug Views: {', '.join(view.capability_debug_views) or 'summary'}",
                        size=11,
                    )
                ]
                + [
                    ft.Text(value=f"{spec.label}: {spec.default_value}", size=12)
                    for spec in view.param_controls
                ],
                spacing=8,
            ),
        )
        content_controls.append(
            ft.Row(controls=[left_panel, center_panel, right_panel], spacing=10)
        )

    return ft.View(
        route=view.route,
        controls=[
            ft.AppBar(title=ft.Text(f"COBA · {view.title}")),
            ft.Container(
                content=ft.Column(
                    controls=content_controls,
                    spacing=10,
                ),
                padding=24,
            ),
        ],
    )


def main(page: Any) -> None:
    """Configure Flet page routing and view rendering."""
    if ft is None:
        raise RuntimeError("Flet is not installed. Install it with `pip install flet`.")

    page.title = "COBA Flet"
    pref_store = PreferencesStore()
    prefs = pref_store.load()

    def save_and_refresh(next_prefs: UserPreferences) -> None:
        pref_store.save(next_prefs)
        page.go(page.route or "/")

    def on_route_change(event: Any) -> None:
        page.views.clear()
        active = build_route_ui_model(event.route, prefs=prefs)
        page.views.append(_render_shell_view(active))
        page.update()

    def on_disconnect(event: Any) -> None:
        _ = event
        save_and_refresh(prefs)

    page.on_route_change = on_route_change
    page.on_disconnect = on_disconnect
    page.go(page.route or "/")


def run() -> None:
    """Start the redesign shell app."""
    if ft is None:
        raise RuntimeError("Flet is not installed. Install it with `pip install flet`.")
    ft.app(target=main)


if __name__ == "__main__":
    run()
