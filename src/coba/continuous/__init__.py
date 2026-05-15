"""Continuous action space bandits using CATS (Continuous Action Tree Sampling).

Reference:
  Wen et al., "Efficient Exploration for Continuous Action Spaces"
  arXiv:1902.01520, ICML 2020.

This module provides real-valued action selection via a binary tree of
LinUCB models. Each tree leaf owns a contextual model; decision time
scores all leaves and samples uniformly within the best leaf's bandwidth.

Typical workflow::

    from coba.continuous import ContinuousBandit

    bandit = ContinuousBandit(
        a_min=0.50,
        a_max=5.00,
        n_features=8,
        depth=6,  # 64 leaves
        alpha=1.0,
    )

    # Train from historical data
    bandit.fit_offline(contexts, actions, rewards, propensities)

    # Serve
    decision = bandit.decide(context)
    # → action ≈ 2.34, propensity ≈ 0.53

    # Update with observed reward
    bandit.update(context, decision.chosen_action, reward, decision.propensity)
"""

from coba.continuous.action_tree import ActionLeaf, BinaryActionTree
from coba.continuous.bandit import ContinuousBandit
from coba.continuous.policy import CATSPolicy
from coba.continuous.schemas import ContinuousDecision

__all__ = [
    "ActionLeaf",
    "BinaryActionTree",
    "ContinuousBandit",
    "CATSPolicy",
    "ContinuousDecision",
]
