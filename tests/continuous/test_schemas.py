"""Tests for continuous action bandit schemas."""

import pytest

from coba.continuous.schemas import ContinuousDecision


class TestContinuousDecision:
    """Test ContinuousDecision validation and creation."""

    def test_valid_decision(self) -> None:
        """ContinuousDecision accepts valid parameters."""
        decision = ContinuousDecision(
            chosen_action=2.5,
            propensity=0.5,
            leaf_index=5,
            leaf_lo=2.0,
            leaf_hi=3.0,
            leaf_scores={0: 1.0, 5: 2.5, 10: 1.2},
            mean_estimate=2.0,
            confidence_width=0.5,
        )
        assert decision.chosen_action == 2.5
        assert decision.propensity == 0.5
        assert decision.leaf_index == 5

    def test_invalid_action_below_bounds(self) -> None:
        """chosen_action below leaf_lo raises ValueError."""
        with pytest.raises(ValueError, match="must be in"):
            ContinuousDecision(
                chosen_action=1.5,
                propensity=0.5,
                leaf_index=5,
                leaf_lo=2.0,
                leaf_hi=3.0,
                leaf_scores={5: 2.5},
                mean_estimate=2.0,
                confidence_width=0.5,
            )

    def test_invalid_action_above_bounds(self) -> None:
        """chosen_action above leaf_hi raises ValueError."""
        with pytest.raises(ValueError, match="must be in"):
            ContinuousDecision(
                chosen_action=3.5,
                propensity=0.5,
                leaf_index=5,
                leaf_lo=2.0,
                leaf_hi=3.0,
                leaf_scores={5: 2.5},
                mean_estimate=2.0,
                confidence_width=0.5,
            )

    def test_invalid_zero_propensity(self) -> None:
        """propensity=0 raises ValueError."""
        with pytest.raises(ValueError, match="propensity must be > 0"):
            ContinuousDecision(
                chosen_action=2.5,
                propensity=0.0,
                leaf_index=5,
                leaf_lo=2.0,
                leaf_hi=3.0,
                leaf_scores={5: 2.5},
                mean_estimate=2.0,
                confidence_width=0.5,
            )

    def test_invalid_negative_propensity(self) -> None:
        """propensity < 0 raises ValueError."""
        with pytest.raises(ValueError, match="propensity must be > 0"):
            ContinuousDecision(
                chosen_action=2.5,
                propensity=-0.1,
                leaf_index=5,
                leaf_lo=2.0,
                leaf_hi=3.0,
                leaf_scores={5: 2.5},
                mean_estimate=2.0,
                confidence_width=0.5,
            )

    def test_leaf_index_not_in_scores(self) -> None:
        """leaf_index not present in leaf_scores raises ValueError."""
        with pytest.raises(ValueError, match="leaf_index .* not in leaf_scores"):
            ContinuousDecision(
                chosen_action=2.5,
                propensity=0.5,
                leaf_index=5,
                leaf_lo=2.0,
                leaf_hi=3.0,
                leaf_scores={0: 1.0, 10: 1.2},
                mean_estimate=2.0,
                confidence_width=0.5,
            )

    def test_boundary_action_at_leaf_lo(self) -> None:
        """chosen_action at leaf_lo is valid."""
        decision = ContinuousDecision(
            chosen_action=2.0,
            propensity=0.5,
            leaf_index=5,
            leaf_lo=2.0,
            leaf_hi=3.0,
            leaf_scores={5: 2.5},
            mean_estimate=2.0,
            confidence_width=0.5,
        )
        assert decision.chosen_action == 2.0

    def test_boundary_action_at_leaf_hi(self) -> None:
        """chosen_action at leaf_hi is valid."""
        decision = ContinuousDecision(
            chosen_action=3.0,
            propensity=0.5,
            leaf_index=5,
            leaf_lo=2.0,
            leaf_hi=3.0,
            leaf_scores={5: 2.5},
            mean_estimate=2.0,
            confidence_width=0.5,
        )
        assert decision.chosen_action == 3.0
