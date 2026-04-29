"""
coba — Bandit Engine for Contextual Bandits

A generic, high-performance contextual bandit library combining:
- contextualbandits: LinUCB/LinTS with Sherman-Morrison updates, off-policy IPS/DR
- mabwiser: Clean architecture, KMeans cluster routing, warm-start, dynamic arm management
"""

from coba.cluster_bandit import ClusterBandit
from coba.schemas import BanditDecision
from coba.types import Arm, PolicyType

__all__ = [
    "ClusterBandit",
    "BanditDecision",
    "Arm",
    "PolicyType",
]
