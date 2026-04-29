"""Tests for Pydantic v2 data models."""



from coba.schemas import BanditDecision


class TestBanditDecision:
    """Tests for BanditDecision."""

    def test_creation(self):
        decision = BanditDecision(
            chosen_arm=1.2,
            score=0.85,
            all_scores={"1.0": 0.5, "1.2": 0.85, "1.5": 0.6},
        )
        assert decision.chosen_arm == 1.2
        assert decision.score == 0.85
