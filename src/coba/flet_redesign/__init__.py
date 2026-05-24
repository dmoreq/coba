"""Flet redesign scaffolding for COBA."""

from coba.flet_redesign.arena import (
    ArenaMetrics,
    ArenaRunStore,
    RunSnapshot,
    SeriesPoint,
    build_arena_metrics,
)
from coba.flet_redesign.contracts import (
    BanditPolicy,
    DebugSnapshotProvider,
    SimulationStepResult,
    World,
)
from coba.flet_redesign.debug import (
    ContextualDebugPane,
    build_linucb_debug_pane,
    build_logistic_debug_pane,
)
from coba.flet_redesign.curriculum import (
    LESSON_REGISTRY,
    LessonConfig,
    LessonObjective,
    LessonProgressState,
    TheoryStageCard,
    evaluate_lesson_objective,
    explain_step_delta,
    get_lesson,
    get_lesson_by_policy,
    locked_control_keys_for_stage,
    render_theory_stage_markdown,
)
from coba.flet_redesign.main import main, run
from coba.flet_redesign.policy_factory import build_policy
from coba.flet_redesign.policies import (
    EpsilonGreedyPolicy,
    LinUCBSWPolicy,
    LinUCBPolicy,
    LogisticUCBPolicy,
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
from coba.flet_redesign.trace import TraceBuffer, filter_trace_records
from coba.flet_redesign.ui import (
    LessonPanelModel,
    PaneSpec,
    PreferencesStore,
    RunControlState,
    RunController,
    ThreePaneLayoutSpec,
    UserPreferences,
    build_three_pane_layout,
)
from coba.flet_redesign.ui.param_controls import ParamControlSpec, default_policy_param_controls
from coba.flet_redesign.ui.tooltips import ParamTooltip
from coba.flet_redesign.ui.view_models import RouteUIModel, build_route_ui_model
from coba.flet_redesign.worlds import (
    ArmDef,
    CONTEXTUAL_PRESETS,
    ConfigurableWorld,
    ContextualPreset,
    FeatureDef,
    WorldConfig,
    create_world,
    get_world_config,
    list_contextual_presets,
    list_world_configs,
)

__all__ = [
    "ArenaMetrics",
    "ArenaRunStore",
    "LESSON_REGISTRY",
    "ArmState",
    "AppRoute",
    "AppSelectionState",
    "AppStateStore",
    "ArmDef",
    "BanditPolicy",
    "CONTEXTUAL_PRESETS",
    "ConfigurableWorld",
    "ContextualDebugPane",
    "ContextualPreset",
    "DiscreteSimulator",
    "DebugSnapshotProvider",
    "EpsilonGreedyPolicy",
    "FeatureDef",
    "LessonConfig",
    "LessonObjective",
    "LessonPanelModel",
    "LessonProgressState",
    "LinUCBSWPolicy",
    "LinUCBPolicy",
    "LogisticUCBPolicy",
    "PaneSpec",
    "ParamControlSpec",
    "ParamTooltip",
    "PreferencesStore",
    "RandomPolicy",
    "RunSnapshot",
    "RouteUIModel",
    "RouteSpec",
    "RunControlState",
    "RunController",
    "RunConfig",
    "ShellView",
    "SeriesPoint",
    "SimulationState",
    "SimulationStepResult",
    "SoftmaxPolicy",
    "TheoryStageCard",
    "ThreePaneLayoutSpec",
    "TraceBuffer",
    "ThompsonSamplingPolicy",
    "UCB1Policy",
    "UserPreferences",
    "WorldConfig",
    "World",
    "build_arena_metrics",
    "build_policy",
    "build_linucb_debug_pane",
    "build_logistic_debug_pane",
    "build_route_ui_model",
    "build_shell_stack",
    "build_three_pane_layout",
    "create_world",
    "default_policy_param_controls",
    "evaluate_lesson_objective",
    "explain_step_delta",
    "filter_trace_records",
    "get_lesson",
    "get_lesson_by_policy",
    "get_world_config",
    "list_world_configs",
    "list_contextual_presets",
    "get_route_spec",
    "list_route_specs",
    "locked_control_keys_for_stage",
    "main",
    "normalize_route",
    "render_theory_stage_markdown",
    "run",
]
