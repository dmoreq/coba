"""coba.simulation — Pure simulation utilities for contextual bandit environments.

This subpackage contains domain-agnostic reward functions and simulation
step primitives. They can be used by standalone simulation scripts,
research notebooks, or service integrations.
"""

from coba.simulation.reward_fns import (
    RewardFn,
    categorical_reward,
    context_free_reward,
    linear_reward,
)

__all__ = [
    "RewardFn",
    "categorical_reward",
    "context_free_reward",
    "linear_reward",
]
