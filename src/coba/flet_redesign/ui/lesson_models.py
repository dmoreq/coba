"""Lesson-specific UI models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LessonPanelModel:
    """View-model for lesson theory/progression panel."""

    lesson_id: str
    lesson_title: str
    stage_index: int
    theory_markdown: str
    locked_controls: tuple[str, ...]
    objective_text: str
    step_explanation: str
