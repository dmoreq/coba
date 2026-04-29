"""

Pydantic v2 data models for the coba bandit system.

These models represent the generic, domain-agnostic interfaces for
interacting with the bandit's decisions and statistics.
"""

from pydantic import BaseModel, Field

from coba.types import Arm


class BanditDecision(BaseModel):
    """Output of the bandit's decide() call."""

    chosen_arm: Arm = Field(..., description="The selected arm identifier")
    score: float = Field(..., description="Score of the chosen arm (e.g. UCB/TS sample)")
    all_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Scores for all arms (arm keys cast to str for JSON compatibility)",
    )

    model_config = {"frozen": True}


class BanditStats(BaseModel):
    """Per-arm statistics reported by the bandit for monitoring."""

    arm: Arm
    n_pulls: int = Field(default=0, description="Number of times arm was selected")
    mean_reward: float = Field(default=0.0)
    last_score: float | None = Field(default=None)

    model_config = {"frozen": False}
