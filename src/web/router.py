"""Route definitions and resolution helpers for the Flet shell."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AppRoute(str, Enum):
    """Supported top-level routes."""

    HOME = "/"
    LESSON = "/lesson"
    ARENA = "/arena"
    SANDBOX = "/sandbox"
    COMPARISON = "/comparison"


@dataclass(frozen=True)
class RouteSpec:
    """Display metadata for one top-level route."""

    route: AppRoute
    title: str
    heading: str
    description: str


_ROUTE_SPECS: dict[AppRoute, RouteSpec] = {
    AppRoute.HOME: RouteSpec(
        route=AppRoute.HOME,
        title="Home",
        heading="COBA Flet Redesign",
        description="Select a destination to start exploring simulations and lessons.",
    ),
    AppRoute.LESSON: RouteSpec(
        route=AppRoute.LESSON,
        title="Lesson",
        heading="Lesson Workspace",
        description="Interactive lesson view with guided objectives and theory cards.",
    ),
    AppRoute.ARENA: RouteSpec(
        route=AppRoute.ARENA,
        title="Arena",
        heading="Arena Workspace",
        description="Run policies in real time and inspect reward/regret behavior.",
    ),
    AppRoute.SANDBOX: RouteSpec(
        route=AppRoute.SANDBOX,
        title="Sandbox",
        heading="Sandbox Workspace",
        description="Experiment with open-ended world and policy parameter settings.",
    ),
    AppRoute.COMPARISON: RouteSpec(
        route=AppRoute.COMPARISON,
        title="Comparison",
        heading="Comparison Workspace",
        description="Run policies side-by-side and compare performance metrics.",
    ),
}


def normalize_route(route: str | None) -> AppRoute:
    """Map a raw path to one of the supported top-level routes."""
    if not route:
        return AppRoute.HOME

    cleaned = route.strip()
    if not cleaned:
        return AppRoute.HOME
    if not cleaned.startswith("/"):
        cleaned = f"/{cleaned}"

    if cleaned == AppRoute.HOME.value:
        return AppRoute.HOME

    for candidate in (AppRoute.LESSON, AppRoute.ARENA, AppRoute.SANDBOX, AppRoute.COMPARISON):
        base = candidate.value
        if cleaned == base or cleaned.startswith(f"{base}/"):
            return candidate

    return AppRoute.HOME


def get_route_spec(route: str | None) -> RouteSpec:
    """Return route metadata for one path."""
    return _ROUTE_SPECS[normalize_route(route)]


def list_route_specs() -> tuple[RouteSpec, ...]:
    """List all route specs in navigation order."""
    return tuple(_ROUTE_SPECS[key] for key in AppRoute)
