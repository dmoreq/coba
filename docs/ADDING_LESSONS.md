# Adding New Lessons to COBA Web

This guide explains how to add a new interactive lesson to the platform (Flet-based web app).

## Overview

A lesson teaches one algorithm through an interactive simulation with 5 staged theory/objective cards that auto-advance.

**Files you'll work with:**
- `src/web/curriculum/lessons.py` — Lesson registry
- `src/web/worlds/core_worlds.py` — Pre-built simulation worlds
- `src/web/policy_factory.py` — Policy instantiation
- `src/web/policy_capabilities.py` — Capability metadata

---

## 1. Define a New Lesson

In `src/web/curriculum/lessons.py`, add a `LessonConfig`:

```python
LessonConfig(
    lesson_id="lesson_my_algo",
    title="My Algorithm",
    policy_id="epsilon_greedy",
    world_id="rural_clinic",
    description="Short description.",
    stages=(
        TheoryStageCard(
            theory="# Stage 1\nExplain the problem in plain English...",
            locked_controls=(),
            objective_text="Run 10 steps.",
            step_explanation="The algorithm chose the best-known arm.",
        ),
        TheoryStageCard(
            theory="# Stage 2\nDeeper dive...",
            locked_controls=("epsilon",),
            objective_text="Experiment with epsilon=0.2.",
            step_explanation="Higher exploration found a better arm.",
        ),
        # ... 5 stages total
    ),
    objective=LessonObjective(min_steps=10, min_cumulative_reward=5.0),
)
```

## 2. Register

Add the config to `LESSON_REGISTRY` in `lessons.py`.
Ensure the policy_id is handled in `build_policy()` in `policy_factory.py`.

## 3. Verify

```bash
uv run pytest tests/flet_redesign/test_curriculum.py -v -p no:asyncio
uv run pytest tests/flet_redesign/test_integration.py -v -p no:asyncio
```
