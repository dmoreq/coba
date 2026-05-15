"""Continuous action space bandits using CATS (Continuous Action Tree Sampling).

Reference:
  Wen et al., "Efficient Exploration for Continuous Action Spaces"
  arXiv:1902.01520, ICML 2020.

This module provides real-valued action selection via a binary tree of
LinUCB models. Each tree leaf owns a contextual model; decision time
scores all leaves and samples uniformly within the best leaf's bandwidth.
"""

from coba.continuous.schemas import ContinuousDecision

__all__ = [
    "ContinuousDecision",
]
