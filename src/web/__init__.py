"""Flet redesign scaffolding for COBA."""

# Core contracts
from web.contracts import (
    BanditPolicy,
    DebugSnapshotProvider,
    SimulationStepResult,
    Simulator,
    World,
)
from web.state import ArmState, RunConfig, SimulationState

# Core engine
from web.simulator import DiscreteSimulator
from web.trace import TraceBuffer, filter_trace_records
from web.drift_monitor import DriftEvent, DriftTimeline

# Worlds
from web.worlds import (
    ArmDef,
    ConfigurableWorld,
    FeatureDef,
    WorldConfig,
    create_world,
    get_world_config,
    list_world_configs,
)
from web.worlds.presets import CONTEXTUAL_PRESETS, ContextualPreset, list_contextual_presets

# Policies
from web.policies import (
    BootstrappedEnsemblePolicy,
    EpsilonGreedyPolicy,
    GPUCBPolicy,
    LinTSPolicy,
    LinUCBHybridPolicy,
    LinUCBSWPolicy,
    LinUCBPolicy,
    LogisticUCBPolicy,
    RandomPolicy,
    SoftmaxPolicy,
    ThompsonSamplingPolicy,
    TreeTSPolicy,
    TreeUCBPolicy,
    UCB1Policy,
)
from web.policy_capabilities import (
    POLICY_CAPABILITIES,
    PolicyCapability,
    get_policy_capability,
)
from web.policy_factory import build_policy

# Routing
from web.router import AppRoute, RouteSpec, get_route_spec, list_route_specs, normalize_route

# Analysis (merged arena + comparison)
from web.analysis import (
    ArenaMetrics,
    ComparisonDiagnostics,
    ComparisonRunResult,
    PolicySummaryStats,
    SeriesPoint,
    build_arena_metrics,
    compute_comparison_diagnostics,
    run_batch_comparison,
    run_policy_comparison,
    summarize_comparison_runs,
)

# UI models (consolidated)
from web.ui.models import (
    ContextInspectionModel,
    DiffViewProps,
    LessonPanelModel,
    ParamTooltip,
    ScenePanelModel,
    SnapshotDiffResult,
    TraceTableModel,
    TreatmentCardModel,
    build_batch_summary_panel,
    build_diff_view_props,
    build_trace_table,
)
from web.ui.charts import ChartData, build_chart_data
from web.ui.layout import PaneSpec, ThreePaneLayoutSpec, build_three_pane_layout
from web.ui.param_controls import ParamControlSpec, default_policy_param_controls
from web.ui.preferences import PreferencesStore, UserPreferences
from web.ui.run_controls import RunControlState, RunController
from web.ui.view_models import RouteUIModel, build_route_ui_model

# Curriculum
from web.curriculum import (
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

# Debug views (cleaned up — context_free.py and continuous.py deleted)
from web.debug import (
    AdvancedDebugPane,
    ContextualDebugPane,
    build_ensemble_debug_pane,
    build_gp_debug_pane,
    build_hybrid_debug_pane,
    build_linucb_debug_pane,
    build_logistic_debug_pane,
    build_tree_debug_pane,
)

# Continuous actions
from web.continuous import (
    CATSLikePolicy,
    ContinuousActionSpace,
    ContinuousSimulator,
    ContinuousStepResult,
    ContinuousWorld,
)

# Entry point
from web.main import main, run


# Re-export all public symbols
__all__ = [
    "AdvancedDebugPane",
    "AppRoute",
    "ArenaMetrics",
    "ArmDef",
    "ArmState",
    "BanditPolicy",
    "BootstrappedEnsemblePolicy",
    "CATSLikePolicy",
    "ChartData",
    "ComparisonDiagnostics",
    "ComparisonRunResult",
    "ConfigurableWorld",
    "CONTEXTUAL_PRESETS",
    "ContextInspectionModel",
    "ContextualDebugPane",
    "ContextualPreset",
    "ContinuousActionSpace",
    "ContinuousSimulator",
    "ContinuousStepResult",
    "ContinuousWorld",
    "DebugSnapshotProvider",
    "DiffViewProps",
    "DiscreteSimulator",
    "DriftEvent",
    "DriftTimeline",
    "EpsilonGreedyPolicy",
    "FeatureDef",
    "GPUCBPolicy",
    "LESSON_REGISTRY",
    "LessonConfig",
    "LessonObjective",
    "LessonPanelModel",
    "LessonProgressState",
    "LinTSPolicy",
    "LinUCBHybridPolicy",
    "LinUCBPolicy",
    "LinUCBSWPolicy",
    "LogisticUCBPolicy",
    "PaneSpec",
    "ParamControlSpec",
    "ParamTooltip",
    "POLICY_CAPABILITIES",
    "PolicyCapability",
    "PolicySummaryStats",
    "PreferencesStore",
    "RandomPolicy",
    "RouteSpec",
    "RouteUIModel",
    "RunConfig",
    "RunControlState",
    "RunController",
    "ScenePanelModel",
    "SeriesPoint",
    "SimulationState",
    "SimulationStepResult",
    "Simulator",
    "SnapshotDiffResult",
    "SoftmaxPolicy",
    "TheoryStageCard",
    "ThompsonSamplingPolicy",
    "ThreePaneLayoutSpec",
    "TraceBuffer",
    "TraceTableModel",
    "TreatmentCardModel",
    "TreeTSPolicy",
    "TreeUCBPolicy",
    "UCB1Policy",
    "UserPreferences",
    "World",
    "WorldConfig",
    "build_arena_metrics",
    "build_batch_summary_panel",
    "build_chart_data",
    "build_diff_view_props",
    "build_policy",
    "build_route_ui_model",
    "build_three_pane_layout",
    "build_trace_table",
    "compute_comparison_diagnostics",
    "create_world",
    "default_policy_param_controls",
    "evaluate_lesson_objective",
    "explain_step_delta",
    "filter_trace_records",
    "get_lesson",
    "get_lesson_by_policy",
    "get_policy_capability",
    "get_route_spec",
    "get_world_config",
    "list_contextual_presets",
    "list_route_specs",
    "list_world_configs",
    "locked_control_keys_for_stage",
    "main",
    "normalize_route",
    "render_theory_stage_markdown",
    "run",
    "run_batch_comparison",
    "run_policy_comparison",
    "summarize_comparison_runs",
    "build_ensemble_debug_pane",
    "build_gp_debug_pane",
    "build_hybrid_debug_pane",
    "build_linucb_debug_pane",
    "build_logistic_debug_pane",
    "build_tree_debug_pane",
]
