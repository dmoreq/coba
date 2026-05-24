"""Pure view-model builders for Flet route rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from coba.flet_redesign.arena import ArenaMetrics, build_arena_metrics
from coba.flet_redesign.curriculum import (
    evaluate_lesson_objective,
    explain_step_delta,
    get_lesson_by_policy,
    locked_control_keys_for_stage,
    render_theory_stage_markdown,
)
from coba.flet_redesign.policies.contextual_utils import context_to_vector
from coba.flet_redesign.router import get_route_spec
from coba.flet_redesign.trace import TraceBuffer
from coba.flet_redesign.ui.components import ScenePanelModel, TreatmentCardModel
from coba.flet_redesign.ui.context_inspection import ContextInspectionModel
from coba.flet_redesign.ui.layout import ThreePaneLayoutSpec, build_three_pane_layout
from coba.flet_redesign.ui.lesson_models import LessonPanelModel
from coba.flet_redesign.ui.param_controls import ParamControlSpec, default_policy_param_controls
from coba.flet_redesign.ui.preferences import UserPreferences
from coba.flet_redesign.worlds import create_world, get_world_config


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


def build_route_ui_model(route: str | None, prefs: UserPreferences) -> RouteUIModel:
    """Build route UI model from preferences and world metadata."""
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
    context = world.sample_context(step_index=1)
    scene_panel = ScenePanelModel(
        world_title=config.title,
        world_description=config.description,
        context_items=context,
    )
    cards = tuple(
        TreatmentCardModel(arm_id=arm.arm_id, label=arm.label, predicted_score=None, selected=False)
        for arm in config.arms
    )
    trace_records = tuple(TraceBuffer().to_records())
    arena_metrics = (
        build_arena_metrics(list(trace_records)) if spec.route.value == "/arena" else None
    )
    contextual_inspection = ContextInspectionModel(
        feature_order=tuple(feature.name for feature in config.features),
        feature_values=tuple(
            context_to_vector(
                context,
                feature_order=tuple(feature.name for feature in config.features),
            )
        ),
        notes="Feature vector preview for contextual policies.",
    )
    lesson_panel = None
    if spec.route.value == "/lesson":
        lesson = get_lesson_by_policy(prefs.policy_id)
        stage = lesson.stages[0]
        lesson_panel = LessonPanelModel(
            lesson_id=lesson.lesson_id,
            lesson_title=lesson.title,
            stage_index=stage.stage_index,
            theory_markdown=render_theory_stage_markdown(stage),
            locked_controls=locked_control_keys_for_stage(lesson, stage=stage.stage_index),
            objective_text=(
                "Objective complete."
                if evaluate_lesson_objective(
                    objective=lesson.objective,
                    steps_executed=0,
                    cumulative_reward=0.0,
                    cumulative_regret=0.0,
                )
                else "Objective pending: run more steps."
            ),
            step_explanation=explain_step_delta(
                previous=None,
                current={
                    "chosen_arm": cards[0].arm_id,
                    "cumulative_reward": 0.0,
                    "cumulative_regret": 0.0,
                },
            ),
        )

    return RouteUIModel(
        route=spec.route.value,
        title=spec.title,
        heading=spec.heading,
        description=spec.description,
        layout=build_three_pane_layout(),
        scene_panel=scene_panel,
        treatment_cards=cards,
        param_controls=default_policy_param_controls(prefs.policy_id),
        trace_records=trace_records,
        arena_metrics=arena_metrics,
        lesson_panel=lesson_panel,
        context_inspection=contextual_inspection,
    )
