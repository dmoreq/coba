"""
coba — Bandit Engine for Contextual Bandits

A generic, high-performance contextual bandit library combining:
- contextualbandits: LinUCB/LinTS with Sherman-Morrison updates, off-policy IPS/DR
- mabwiser: Clean architecture, KMeans cluster routing, warm-start, dynamic arm management
"""

from coba.bandit import ClusterBandit
from coba.drift import PageHinkleyDetector
from coba.normalizer import RewardNormalizer
from coba.schemas import BanditDecision, ScoreBreakdown
from coba.types import Arm, PolicyType

__all__ = [
    "ClusterBandit",
    "BanditDecision",
    "ScoreBreakdown",
    "PageHinkleyDetector",
    "RewardNormalizer",
    "Arm",
    "PolicyType",
]
