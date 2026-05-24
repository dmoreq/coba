"""Flet redesign scaffolding for COBA."""

from coba.flet_redesign.contracts import (
    BanditPolicy,
    DebugSnapshotProvider,
    SimulationStepResult,
    World,
)
from coba.flet_redesign.main import main, run
from coba.flet_redesign.policies import (
    EpsilonGreedyPolicy,
    RandomPolicy,
    SoftmaxPolicy,
    ThompsonSamplingPolicy,
    UCB1Policy,
)
from coba.flet_redesign.router import (
    AppRoute,
    RouteSpec,
    get_route_spec,
    list_route_specs,
    normalize_route,
)
from coba.flet_redesign.shell import ShellView, build_shell_stack
from coba.flet_redesign.simulator import DiscreteSimulator
from coba.flet_redesign.state_store import AppSelectionState, AppStateStore
from coba.flet_redesign.state import ArmState, RunConfig, SimulationState
from coba.flet_redesign.trace import TraceBuffer
from coba.flet_redesign.worlds import (
    ArmDef,
    ConfigurableWorld,
    FeatureDef,
    WorldConfig,
    create_world,
    get_world_config,
    list_world_configs,
)

__all__ = [
    "ArmState",
    "AppRoute",
    "AppSelectionState",
    "AppStateStore",
    "ArmDef",
    "BanditPolicy",
    "ConfigurableWorld",
    "DiscreteSimulator",
    "DebugSnapshotProvider",
    "EpsilonGreedyPolicy",
    "FeatureDef",
    "RandomPolicy",
    "RouteSpec",
    "RunConfig",
    "ShellView",
    "SimulationState",
    "SimulationStepResult",
    "SoftmaxPolicy",
    "TraceBuffer",
    "ThompsonSamplingPolicy",
    "UCB1Policy",
    "WorldConfig",
    "World",
    "build_shell_stack",
    "create_world",
    "get_world_config",
    "list_world_configs",
    "get_route_spec",
    "list_route_specs",
    "main",
    "normalize_route",
    "run",
]
