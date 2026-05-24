# Adding a World

A world defines a scenario with features, arms, and a reward model. Follow these steps to add a new narrative world.

## 1. Define the World Config

In `src/web/worlds/core_worlds.py`, add a new `WorldConfig` constant:

```python
MY_WORLD = WorldConfig(
    world_id="my_world",
    title="My World",
    description="Short description of the scenario.",
    difficulty="easy",  # easy, medium, or hard
    features=(
        FeatureDef(name="feature_1", feature_type="numeric", numeric_min=0.0, numeric_max=1.0),
        FeatureDef(name="feature_2", feature_type="binary"),
        FeatureDef(
            name="feature_3",
            feature_type="categorical",
            categories=("a", "b", "c"),
        ),
    ),
    arms=(
        ArmDef(
            arm_id="arm_a",
            label="Arm A",
            base_rate=0.5,
            weights={"feature_1": 0.1, "feature_2": 0.2, "feature_3": 0.05},
        ),
        # ... add 2-3 arms total
    ),
)
```

**Feature types:**
- `numeric`: continuous value in `[numeric_min, numeric_max]`
- `binary`: 0 or 1
- `categorical`: one of the listed categories

**Arm weights:** Each arm has a `base_rate` (0.0–1.0) and per-feature weights. The reward is Bernoulli with probability = `sigmoid(logit(base_rate) + Σ weight × normalized_feature_value)`. Higher weights make an arm more effective for high feature values.

## 2. Register the World

Add your world to `CORE_WORLD_CONFIGS` at the bottom of `core_worlds.py`:

```python
CORE_WORLD_CONFIGS: tuple[WorldConfig, ...] = (
    RURAL_CLINIC_WORLD,
    # ... existing worlds ...
    MY_WORLD,
)
```

The registry in `worlds/registry.py` auto-discovers from this tuple — no other registration needed.

## 3. Add Fixture Data

Update `tests/flet_redesign/fixtures/core_world_fixtures.json`:

```json
{"world_id": "my_world", "n_features": 3, "n_arms": 3}
```

## 4. Add a Lesson

In `src/web/curriculum/lessons.py`, create a lesson that uses your world:

```python
"lesson_my_world": LessonConfig(
    lesson_id="lesson_my_world",
    title="My World with UCB1",
    policy_id="ucb1",
    world_id="my_world",
    stages=_stages_for("UCB1", "score = mean + alpha * sqrt(2*log(t)/n)"),
    objective=LessonObjective(
        min_steps=80, min_cumulative_reward=38.0, max_cumulative_regret=42.0
    ),
    stage_locked_controls={
        1: ("alpha",),
        2: ("seed",),
        3: ("seed",),
        4: ("seed",),
        5: (),
    },
),
```

## 5. Verify

```bash
PYTHONPATH=src python -c "
from web.worlds import create_world, list_world_configs
configs = list_world_configs()
print(f'{len(configs)} worlds')
w = create_world('my_world')
ctx = w.sample_context(1)
r = w.sample_reward(ctx, 'arm_a')
print(f'Reward: {r}')
"
pytest tests/flet_redesign -q -k test_worlds
```

## Tips

- Start with `difficulty="easy"` and moderate base rates (0.45–0.58) for balanced arms.
- Ensure at least one arm gets better with high feature values and one gets worse — this creates interesting learning dynamics.
- Test with `random`, `ucb1`, and `linucb` policies to verify the world works across algorithm families.
