"""coba.simulation — Pure simulation utilities for contextual bandit environments.

This subpackage contains domain-agnostic reward functions and simulation
step primitives that have no dependency on any web framework. They can be
used both by the coba-web Dash application and by any standalone simulation
script or notebook.
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
