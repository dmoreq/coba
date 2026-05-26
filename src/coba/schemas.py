"""

Pydantic v2 data models for the coba bandit system.

These models represent the generic, domain-agnostic interfaces for
interacting with the bandit's decisions and statistics.
"""

from pydantic import BaseModel, Field

from coba.types import Arm


class ScoreBreakdown(BaseModel):
    """Per-arm decomposition of a decision score."""

    score: float = Field(..., description="Final arm score used for ranking.")
    mean_estimate: float | None = Field(
        default=None,
        description="Expected reward / exploitation component when available.",
    )
    confidence_width: float | None = Field(
        default=None,
        description="Exploration bonus / uncertainty component when available.",
    )
    is_fitted: bool = Field(default=False, description="Whether the routed arm model is fitted.")
    n_obs: int | None = Field(
        default=None, description="Observations seen by the routed arm model."
    )

    model_config = {"frozen": True}


class BanditDecision(BaseModel):
    """Output of the bandit's decide() call."""

    chosen_arm: Arm | None = Field(
        ...,
        description="The selected arm identifier. None when the decision was abstained.",
    )
    score: float = Field(..., description="Score of the chosen arm (e.g. UCB/TS sample)")
    all_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Scores for all arms (arm keys cast to str for JSON compatibility)",
    )
    abstained: bool = Field(
        default=False,
        description=(
            "True when no arm had a sufficiently clear lead over the others "
            "(controlled by min_confidence_gap in decide())."
        ),
    )
    # Optional decomposition of score into exploitation + exploration terms.
    # Populated by LinUCB (and future UCB variants) for observability.
    mean_estimate: float | None = Field(
        default=None,
        description="Expected reward component of the score (exploitation term, x @ beta).",
    )
    confidence_width: float | None = Field(
        default=None,
        description="Exploration bonus component of the score (UCB width, alpha * sqrt(x A_inv x)).",
    )

    score_breakdown: dict[str, ScoreBreakdown] = Field(
        default_factory=dict,
        description="Per-arm score decomposition keyed by stringified arm identifier.",
    )

    was_random: bool = Field(
        default=False,
        description=(
            "True when the arm was selected by random exploration "
            "(epsilon-greedy policy) rather than by exploitation."
        ),
    )

    model_config = {"frozen": True}


class BanditStats(BaseModel):
    """Per-arm statistics reported by the bandit for monitoring."""

    arm: Arm
    n_pulls: int = Field(default=0, description="Number of times arm was selected")
    mean_reward: float = Field(default=0.0)
    last_score: float | None = Field(default=None)

    model_config = {"frozen": False}

    def record(self, reward: float) -> None:
        """Increment pull count and update running mean using Welford's algorithm.

        Welford's online update avoids floating-point accumulation drift that
        occurs when summing large numbers of rewards:
            mean_n = mean_{n-1} + (x - mean_{n-1}) / n
        """
        self.n_pulls += 1
        self.mean_reward += (reward - self.mean_reward) / self.n_pulls
