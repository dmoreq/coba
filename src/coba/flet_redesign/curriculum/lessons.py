"""Lesson metadata, theory cards, objectives, and progression helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TheoryStageCard:
    """One stage in the five-stage theory pedagogy."""

    stage_index: int
    title: str
    intuition: str
    formula: str
    practical_hint: str


@dataclass(frozen=True)
class LessonObjective:
    """Objective gate for lesson completion."""

    min_steps: int
    min_cumulative_reward: float
    max_cumulative_regret: float


@dataclass(frozen=True)
class LessonConfig:
    """Lesson metadata."""

    lesson_id: str
    title: str
    policy_id: str
    world_id: str
    stages: tuple[TheoryStageCard, ...]
    objective: LessonObjective
    stage_locked_controls: dict[int, tuple[str, ...]]


@dataclass(frozen=True)
class LessonProgressState:
    """Current lesson progression state."""

    lesson_id: str
    current_stage: int = 1
    completed: bool = False

    def advance(self) -> LessonProgressState:
        if self.current_stage >= 5:
            return LessonProgressState(
                lesson_id=self.lesson_id,
                current_stage=5,
                completed=self.completed,
            )
        return LessonProgressState(
            lesson_id=self.lesson_id,
            current_stage=self.current_stage + 1,
            completed=self.completed,
        )

    def mark_completed(self) -> LessonProgressState:
        return LessonProgressState(
            lesson_id=self.lesson_id,
            current_stage=self.current_stage,
            completed=True,
        )


def _stages_for(policy_name: str, formula: str) -> tuple[TheoryStageCard, ...]:
    return (
        TheoryStageCard(
            stage_index=1,
            title="Problem Framing",
            intuition=f"Why {policy_name} is useful in uncertain decision loops.",
            formula="Define reward objective and sequential decision process.",
            practical_hint="Focus on balancing exploration and exploitation.",
        ),
        TheoryStageCard(
            stage_index=2,
            title="Decision Rule",
            intuition=f"How {policy_name} selects one arm at each step.",
            formula=formula,
            practical_hint="Inspect per-arm scores before each action.",
        ),
        TheoryStageCard(
            stage_index=3,
            title="Update Rule",
            intuition="How observations update internal beliefs.",
            formula="state <- update(state, context, arm, reward)",
            practical_hint="Track when estimates overreact or lag.",
        ),
        TheoryStageCard(
            stage_index=4,
            title="Failure Modes",
            intuition="Common pitfalls under sparse or drifting rewards.",
            formula="regret_t = optimal_reward_t - observed_reward_t",
            practical_hint="Tune parameters to reduce unstable oscillations.",
        ),
        TheoryStageCard(
            stage_index=5,
            title="Operational Use",
            intuition="Translate policy behavior into practical deployment checks.",
            formula="monitor reward, regret, pull share, and uncertainty",
            practical_hint="Validate against deterministic seed replay before rollout.",
        ),
    )


LESSON_REGISTRY: dict[str, LessonConfig] = {
    "lesson_random_baseline": LessonConfig(
        lesson_id="lesson_random_baseline",
        title="Random Baseline",
        policy_id="random",
        world_id="rural_clinic",
        stages=_stages_for("Random Policy", "a_t ~ Uniform(arms)"),
        objective=LessonObjective(
            min_steps=60, min_cumulative_reward=20.0, max_cumulative_regret=40.0
        ),
        stage_locked_controls={
            1: ("seed",),
            2: ("seed", "speed"),
            3: ("seed", "speed"),
            4: ("seed",),
            5: (),
        },
    ),
    "lesson_epsilon_greedy": LessonConfig(
        lesson_id="lesson_epsilon_greedy",
        title="Epsilon-Greedy",
        policy_id="epsilon_greedy",
        world_id="moviematch",
        stages=_stages_for(
            "Epsilon-Greedy", "a_t = random with p=epsilon else argmax(mean_reward)"
        ),
        objective=LessonObjective(
            min_steps=80, min_cumulative_reward=36.0, max_cumulative_regret=40.0
        ),
        stage_locked_controls={
            1: ("epsilon", "seed"),
            2: ("seed",),
            3: ("seed",),
            4: ("seed",),
            5: (),
        },
    ),
    "lesson_ucb1": LessonConfig(
        lesson_id="lesson_ucb1",
        title="UCB1",
        policy_id="ucb1",
        world_id="rural_clinic",
        stages=_stages_for("UCB1", "score = mean + alpha*sqrt(2*log(t)/n)"),
        objective=LessonObjective(
            min_steps=80, min_cumulative_reward=40.0, max_cumulative_regret=35.0
        ),
        stage_locked_controls={
            1: ("alpha", "seed"),
            2: ("seed",),
            3: ("seed",),
            4: ("seed",),
            5: (),
        },
    ),
    "lesson_thompson_sampling": LessonConfig(
        lesson_id="lesson_thompson_sampling",
        title="Thompson Sampling",
        policy_id="thompson",
        world_id="newsfeed",
        stages=_stages_for(
            "Thompson Sampling", "theta_a ~ Beta(alpha_a, beta_a); pick argmax(theta_a)"
        ),
        objective=LessonObjective(
            min_steps=80, min_cumulative_reward=40.0, max_cumulative_regret=35.0
        ),
        stage_locked_controls={
            1: ("prior_alpha", "prior_beta", "seed"),
            2: ("seed",),
            3: ("seed",),
            4: ("seed",),
            5: (),
        },
    ),
    "lesson_softmax": LessonConfig(
        lesson_id="lesson_softmax",
        title="Softmax Exploration",
        policy_id="softmax",
        world_id="moviematch",
        stages=_stages_for("Softmax", "P(a)=exp(Q(a)/tau) / Σ exp(Q(i)/tau)"),
        objective=LessonObjective(
            min_steps=80, min_cumulative_reward=34.0, max_cumulative_regret=42.0
        ),
        stage_locked_controls={
            1: ("tau", "seed"),
            2: ("seed",),
            3: ("seed",),
            4: ("seed",),
            5: (),
        },
    ),
}


def get_lesson(lesson_id: str) -> LessonConfig:
    try:
        return LESSON_REGISTRY[lesson_id]
    except KeyError as exc:
        raise KeyError(f"Unknown lesson_id '{lesson_id}'") from exc


def render_theory_stage_markdown(stage: TheoryStageCard) -> str:
    """Render one stage into markdown."""
    return "\n".join(
        [
            f"### Stage {stage.stage_index}: {stage.title}",
            "",
            f"- Intuition: {stage.intuition}",
            f"- Formula: `{stage.formula}`",
            f"- Practice: {stage.practical_hint}",
        ]
    )


def evaluate_lesson_objective(
    *,
    objective: LessonObjective,
    steps_executed: int,
    cumulative_reward: float,
    cumulative_regret: float,
) -> bool:
    """Evaluate whether lesson objective is satisfied."""
    return (
        steps_executed >= objective.min_steps
        and cumulative_reward >= objective.min_cumulative_reward
        and cumulative_regret <= objective.max_cumulative_regret
    )


def locked_control_keys_for_stage(config: LessonConfig, stage: int) -> tuple[str, ...]:
    """Return locked controls for one theory stage."""
    return config.stage_locked_controls.get(stage, ())


def explain_step_delta(previous: dict[str, Any] | None, current: dict[str, Any]) -> str:
    """Generate a concise explanation from two adjacent trace records."""
    if previous is None:
        return "Initial step: baseline reward/regret established."
    reward_delta = float(current["cumulative_reward"]) - float(previous["cumulative_reward"])
    regret_delta = float(current["cumulative_regret"]) - float(previous["cumulative_regret"])
    arm = current.get("chosen_arm")
    return (
        f"Chose arm '{arm}'. Reward delta: {reward_delta:.3f}. "
        f"Regret delta: {regret_delta:.3f}. "
        "Use this to inspect whether exploration is paying off."
    )
