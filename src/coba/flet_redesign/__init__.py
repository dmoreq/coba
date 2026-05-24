"""Flet redesign scaffolding for COBA."""

from coba.flet_redesign.contracts import (
    BanditPolicy,
    DebugSnapshotProvider,
    SimulationStepResult,
    World,
)
from coba.flet_redesign.main import main, run
from coba.flet_redesign.router import (
    AppRoute,
    RouteSpec,
    get_route_spec,
    list_route_specs,
    normalize_route,
)
from coba.flet_redesign.shell import ShellView, build_shell_stack
from coba.flet_redesign.state import ArmState, RunConfig, SimulationState

__all__ = [
    "ArmState",
    "AppRoute",
    "BanditPolicy",
    "DebugSnapshotProvider",
    "RouteSpec",
    "RunConfig",
    "ShellView",
    "SimulationState",
    "SimulationStepResult",
    "World",
    "build_shell_stack",
    "get_route_spec",
    "list_route_specs",
    "main",
    "normalize_route",
    "run",
]
