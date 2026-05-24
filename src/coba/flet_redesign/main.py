"""Flet app entrypoint for the redesign shell."""

from __future__ import annotations

from typing import Any

from coba.flet_redesign.shell import build_shell_stack

try:
    import flet as ft
except ModuleNotFoundError:  # pragma: no cover - exercised by tests via run()
    ft = None  # type: ignore[assignment]


def _render_shell_view(view: Any) -> Any:
    """Render one shell view using Flet controls."""
    if ft is None:
        raise RuntimeError("Flet is not installed. Install it with `pip install flet`.")

    return ft.View(
        route=view.route,
        controls=[
            ft.AppBar(title=ft.Text(f"COBA · {view.title}")),
            ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Text(value=view.heading, size=28, weight=ft.FontWeight.BOLD),
                        ft.Text(value=view.description, size=14),
                    ],
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

    def on_route_change(event: Any) -> None:
        page.views.clear()
        for shell_view in build_shell_stack(event.route):
            page.views.append(_render_shell_view(shell_view))
        page.update()

    page.on_route_change = on_route_change
    page.go(page.route or "/")


def run() -> None:
    """Start the redesign shell app."""
    if ft is None:
        raise RuntimeError("Flet is not installed. Install it with `pip install flet`.")
    ft.app(target=main)


if __name__ == "__main__":
    run()
