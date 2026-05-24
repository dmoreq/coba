"""Tests for foundation curriculum models and progression."""

from __future__ import annotations

from coba.flet_redesign.curriculum import (
    LESSON_REGISTRY,
    LessonProgressState,
    evaluate_lesson_objective,
    explain_step_delta,
    get_lesson,
    locked_control_keys_for_stage,
    render_theory_stage_markdown,
)


def test_foundation_registry_contains_first_five_lessons() -> None:
    assert len(LESSON_REGISTRY) >= 5
    expected_ids = {
        "lesson_random_baseline",
        "lesson_epsilon_greedy",
        "lesson_ucb1",
        "lesson_thompson_sampling",
        "lesson_softmax",
    }
    assert expected_ids.issubset(set(LESSON_REGISTRY.keys()))


def test_all_lessons_have_five_theory_stages() -> None:
    for lesson in LESSON_REGISTRY.values():
        assert len(lesson.stages) == 5
        assert [stage.stage_index for stage in lesson.stages] == [1, 2, 3, 4, 5]


def test_theory_stage_renderer_outputs_markdown() -> None:
    lesson = get_lesson("lesson_ucb1")
    output = render_theory_stage_markdown(lesson.stages[0])
    assert output.startswith("### Stage 1")
    assert "Formula" in output


def test_objective_evaluation_and_completion_flow() -> None:
    lesson = get_lesson("lesson_ucb1")
    assert evaluate_lesson_objective(
        objective=lesson.objective,
        steps_executed=lesson.objective.min_steps,
        cumulative_reward=lesson.objective.min_cumulative_reward,
        cumulative_regret=lesson.objective.max_cumulative_regret,
    )
    assert not evaluate_lesson_objective(
        objective=lesson.objective,
        steps_executed=lesson.objective.min_steps - 1,
        cumulative_reward=lesson.objective.min_cumulative_reward,
        cumulative_regret=lesson.objective.max_cumulative_regret,
    )

    progress = LessonProgressState(lesson_id=lesson.lesson_id)
    for _ in range(4):
        progress = progress.advance()
    assert progress.current_stage == 5
    completed = progress.mark_completed()
    assert completed.completed


def test_locked_controls_and_step_explanation() -> None:
    lesson = get_lesson("lesson_softmax")
    locked = locked_control_keys_for_stage(lesson, stage=1)
    assert "tau" in locked

    explanation = explain_step_delta(
        previous={"cumulative_reward": 1.0, "cumulative_regret": 0.5},
        current={"chosen_arm": "a", "cumulative_reward": 1.6, "cumulative_regret": 0.7},
    )
    assert "Reward delta" in explanation
    assert "Regret delta" in explanation
