"""Flet-agnostic shell composition for the redesign app."""

from __future__ import annotations

from dataclasses import dataclass

from coba.flet_redesign.router import AppRoute, RouteSpec, get_route_spec


@dataclass(frozen=True)
class ShellView:
    """Minimal view payload that can be rendered by a UI adapter."""

    route: str
    title: str
    heading: str
    description: str


def to_shell_view(spec: RouteSpec) -> ShellView:
    """Convert route metadata into a renderable shell view payload."""
    return ShellView(
        route=spec.route.value,
        title=spec.title,
        heading=spec.heading,
        description=spec.description,
    )


def build_shell_stack(route: str | None) -> list[ShellView]:
    """Build a deterministic view stack from one route.

    Home is always the stack root. For non-home routes we push both:
    1) Home root view
    2) Active route view
    """
    active_spec = get_route_spec(route)
    home_view = to_shell_view(get_route_spec(AppRoute.HOME.value))
    if active_spec.route == AppRoute.HOME:
        return [home_view]
    return [home_view, to_shell_view(active_spec)]
