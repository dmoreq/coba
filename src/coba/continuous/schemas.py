"""Dataclasses for continuous action bandit decisions and events."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContinuousDecision:
    """Result of a continuous action selection decision.

    Attributes:
        chosen_action: The selected real-valued action (e.g., bid price in USD).
        propensity: Probability density p(action | context) under the current policy.
                   For CATS: propensity = 1 / effective_window_width.
                   Used for off-policy evaluation and IPS correction.
        leaf_index: Which tree leaf was selected (0 to n_leaves-1).
        leaf_lo: Lower bound of the winning leaf's action interval.
        leaf_hi: Upper bound of the winning leaf's action interval.
        leaf_scores: Dictionary mapping all leaf indices to their LinUCB scores.
                    Useful for monitoring which action regions are promising.
        mean_estimate: Exploitation term of the winning leaf's score
                      (the predicted reward without exploration bonus).
        confidence_width: Exploration term of the winning leaf's score
                         (the ±UCB width).
    """

    chosen_action: float
    propensity: float
    leaf_index: int
    leaf_lo: float
    leaf_hi: float
    leaf_scores: dict[int, float]
    mean_estimate: float
    confidence_width: float

    def __post_init__(self) -> None:
        """Validate decision fields."""
        if not (self.leaf_lo <= self.chosen_action <= self.leaf_hi):
            raise ValueError(
                f"chosen_action {self.chosen_action} must be in [{self.leaf_lo}, {self.leaf_hi}]"
            )
        if self.propensity <= 0:
            raise ValueError(f"propensity must be > 0, got {self.propensity}")
        if self.leaf_index not in self.leaf_scores:
            raise ValueError(
                f"leaf_index {self.leaf_index} not in leaf_scores keys: "
                f"{list(self.leaf_scores.keys())}"
            )
