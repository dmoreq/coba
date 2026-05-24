# Adding a Lesson

Lessons pair a policy with a world and guide users through 5 theory stages with interactive objectives.

## 1. Lesson Structure

Each lesson in `src/web/curriculum/lessons.py` follows this pattern:

```python
"lesson_id": LessonConfig(
    lesson_id="lesson_id",
    title="Lesson Title",
    policy_id="policy_id",      # must be registered in policy_factory
    world_id="world_id",        # must be registered in CORE_WORLD_CONFIGS
    stages=_stages_for("Algorithm Name", "formula string"),
    objective=LessonObjective(
        min_steps=80,            # minimum steps before evaluation
        min_cumulative_reward=40.0,
        max_cumulative_regret=40.0,
    ),
    stage_locked_controls={
        1: ("alpha", "seed"),    # controls locked at each stage
        2: ("seed",),
        3: ("seed",),
        4: ("seed",),
        5: (),                   # stage 5: all controls unlocked
    },
),
```

## 2. Theory Stages

The `_stages_for()` helper generates 5 stages:

| Stage | Title | Focus |
|---|---|---|
| 1 | Problem Framing | Why this algorithm is useful |
| 2 | Decision Rule | How the algorithm selects arms |
| 3 | Update Rule | How observations update beliefs |
| 4 | Failure Modes | Common pitfalls |
| 5 | Operational Use | Practical deployment |

To create custom stages (not using the template), build a tuple of `TheoryStageCard` directly:

```python
stages=(
    TheoryStageCard(
        stage_index=1,
        title="Custom Title",
        intuition="Why this matters.",
        formula="mathematical formula",
        practical_hint="Tip for users.",
    ),
    # ... stages 2-5
),
```

## 3. Setting Objectives

Choose thresholds that are achievable within ~80-140 steps for the chosen policy-world pair. Test empirically:

```python
from web.policy_factory import build_policy
from web.simulator import DiscreteSimulator
from web.state import RunConfig
from web.worlds import create_world, get_world_config

world = create_world("your_world")
fo = tuple(f.name for f in get_world_config("your_world").features)
policy = build_policy("your_policy", feature_order=fo, seed=0)
sim = DiscreteSimulator(policy, world, RunConfig(seed=0, horizon=200))
sim.reset()
sim.run_steps(140)
print(f"Reward: {sim.state.cumulative_reward}, Regret: {sim.state.cumulative_regret}")
# Set min_cumulative_reward slightly below this, max_cumulative_regret slightly above
```

## 4. Stage-Locked Controls

`stage_locked_controls` maps stage index → tuple of control keys to lock. This enforces progressive disclosure: early stages restrict parameter access.

Common pattern:
- Stage 1: lock key parameters (force exploration of defaults)
- Stages 2-4: lock seed only (focus on algorithm behavior)
- Stage 5: no locks (free experimentation)

## 5. Register the Lesson

Add your lesson to `LESSON_REGISTRY` in `curriculum/lessons.py`. All lessons are registered in a flat dict keyed by `lesson_id`.

## 6. Verify

```bash
PYTHONPATH=src python -c "
from web.curriculum import get_lesson, evaluate_lesson_objective
from web.policy_factory import build_policy
from web.simulator import DiscreteSimulator
from web.state import RunConfig
from web.worlds import create_world, get_world_config

lesson = get_lesson('lesson_id')
fo = tuple(f.name for f in get_world_config(lesson.world_id).features)
p = build_policy(lesson.policy_id, feature_order=fo, seed=0)
w = create_world(lesson.world_id)
sim = DiscreteSimulator(p, w, RunConfig(seed=0, horizon=500))
sim.reset()
sim.run_steps(lesson.objective.min_steps)
print(evaluate_lesson_objective(
    objective=lesson.objective,
    steps_executed=sim.state.current_step,
    cumulative_reward=sim.state.cumulative_reward,
    cumulative_regret=sim.state.cumulative_regret,
))  # Should be True
"
pytest tests/flet_redesign -q -k test_lesson
```

## Tips

- Start with `min_steps=80`, verify, then adjust upward if objectives aren't met deterministically.
- Use fixed seeds (`seed=0`) when tuning objectives to ensure reproducibility.
- Each policy should have at least one lesson. Context-free policies pair with any world; contextual policies need worlds with features.
