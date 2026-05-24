"""Pure view-model builders for Flet route rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from web.arena import ArenaMetrics, build_arena_metrics
from web.curriculum import (
    LessonConfig,
    LessonProgressState,
    evaluate_lesson_objective,
    explain_step_delta,
    get_lesson_by_policy,
    locked_control_keys_for_stage,
    render_theory_stage_markdown,
)
from web.policy_capabilities import get_policy_capability
from web.policies.contextual_utils import context_to_vector
from web.router import get_route_spec
from web.trace import TraceBuffer
from web.ui.components import ScenePanelModel, TreatmentCardModel
from web.ui.context_inspection import ContextInspectionModel
from web.ui.layout import ThreePaneLayoutSpec, build_three_pane_layout
from web.ui.lesson_models import LessonPanelModel
from web.ui.param_controls import ParamControlSpec, default_policy_param_controls
from web.ui.preferences import UserPreferences
from web.worlds import create_world, get_world_config


@dataclass(frozen=True)
class RouteUIModel:
    """Top-level view-model for one route."""

    route: str
    title: str
    heading: str
    description: str
    layout: ThreePaneLayoutSpec | None
    scene_panel: ScenePanelModel | None = None
    treatment_cards: tuple[TreatmentCardModel, ...] = ()
    param_controls: tuple[ParamControlSpec, ...] = ()
    run_control_labels: tuple[str, ...] = field(
        default_factory=lambda: ("Step", "Play", "Pause", "Reset")
    )
    trace_records: tuple[dict[str, Any], ...] = ()
    arena_metrics: ArenaMetrics | None = None
    lesson_panel: LessonPanelModel | None = None
    context_inspection: ContextInspectionModel | None = None
    capability_debug_views: tuple[str, ...] = ()
    locked_controls: tuple[str, ...] = ()


def build_route_ui_model(
    route: str | None,
    prefs: UserPreferences,
    *,
    trace_records: tuple[dict[str, Any], ...] | None = None,
    sim_context: dict[str, Any] | None = None,
    lesson_progress: LessonProgressState | None = None,
    lesson_config: LessonConfig | None = None,
    sim_step_index: int = 0,
    sim_cumulative_reward: float = 0.0,
    sim_cumulative_regret: float = 0.0,
    disabled_control_keys: tuple[str, ...] | None = None,
) -> RouteUIModel:
    """Build route UI model from preferences, optionally overlaid with live simulation state."""
    spec = get_route_spec(route)
    if spec.route.value == "/":
        return RouteUIModel(
            route=spec.route.value,
            title=spec.title,
            heading=spec.heading,
            description=spec.description,
            layout=None,
        )

    config = get_world_config(prefs.world_id)
    world = create_world(prefs.world_id)
    world.reset(seed=0)
    context = sim_context if sim_context else world.sample_context(step_index=1)
    scene_panel = ScenePanelModel(
        world_title=config.title,
        world_description=config.description,
        context_items=context,
    )

    is_lesson_route = spec.route.value.startswith("/lesson")
    if is_lesson_route and lesson_config is None:
        lesson_config = get_lesson_by_policy(prefs.policy_id)

    raw_records = list(trace_records) if trace_records else []
    cards = tuple(
        TreatmentCardModel(
            arm_id=arm.arm_id,
            label=arm.label,
            predicted_score=None,
            selected=(raw_records[-1].get("chosen_arm") == arm.arm_id if raw_records else False),
        )
        for arm in config.arms
    )

    trace_data = trace_records if trace_records else tuple(TraceBuffer().to_records())
    is_arena = spec.route.value == "/arena"
    arena_metrics = build_arena_metrics(raw_records) if is_arena else None

    contextual_inspection = ContextInspectionModel(
        feature_order=tuple(feature.name for feature in config.features),
        feature_values=tuple(
            context_to_vector(
                context,
                feature_order=tuple(feature.name for feature in config.features),
            )
        ),
        notes="Live feature vector.",
    )
    capability = get_policy_capability(prefs.policy_id)

    lesson_panel = None
    locked_controls: tuple[str, ...] = ()
    if is_lesson_route and lesson_config is not None:
        progress = lesson_progress or LessonProgressState(
            lesson_id=lesson_config.lesson_id, current_stage=1, completed=False
        )
        stage_idx = progress.current_stage
        stage = lesson_config.stages[stage_idx - 1]
        obj_met = evaluate_lesson_objective(
            objective=lesson_config.objective,
            steps_executed=sim_step_index,
            cumulative_reward=sim_cumulative_reward,
            cumulative_regret=sim_cumulative_regret,
        )
        prev_record = raw_records[-2] if len(raw_records) >= 2 else None
        current_record = (
            raw_records[-1]
            if raw_records
            else {"chosen_arm": "n/a", "cumulative_reward": 0.0, "cumulative_regret": 0.0}
        )

        lesson_panel = LessonPanelModel(
            lesson_id=lesson_config.lesson_id,
            lesson_title=lesson_config.title,
            stage_index=stage_idx,
            theory_markdown=render_theory_stage_markdown(stage),
            locked_controls=locked_control_keys_for_stage(lesson_config, stage=stage_idx),
            objective_text=(
                "Objective complete."
                if obj_met
                else f"Run more steps. Reward: {sim_cumulative_reward:.1f} / {lesson_config.objective.min_cumulative_reward:.0f}"
            ),
            step_explanation=explain_step_delta(previous=prev_record, current=current_record),
        )
        locked_controls = locked_control_keys_for_stage(lesson_config, stage=stage_idx)

    if disabled_control_keys is not None:
        locked_controls = disabled_control_keys

    return RouteUIModel(
        route=spec.route.value,
        title=spec.title,
        heading=spec.heading,
        description=spec.description,
        layout=build_three_pane_layout(),
        scene_panel=scene_panel,
        treatment_cards=cards,
        param_controls=default_policy_param_controls(prefs.policy_id),
        trace_records=trace_data,
        arena_metrics=arena_metrics,
        lesson_panel=lesson_panel,
        context_inspection=contextual_inspection,
        capability_debug_views=capability.debug_views,
        locked_controls=locked_controls,
    )
