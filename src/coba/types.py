"""
Common type aliases used throughout the coba library.

Defines lightweight semantic aliases so that function signatures
are self-documenting without requiring heavy class overhead.
"""

from enum import Enum

# An arm is any hashable identifier for a choice (e.g. 1.0, "1.2x", 3, "arm_A")
Arm = str | int | float

# A generic numeric type used in reward and score calculations
Num = int | float


class PolicyType(str, Enum):
    """Available bandit learning policies."""

    LIN_UCB = "linucb"
    LIN_TS = "lints"
    LIN_UCB_HYBRID = "linucb_hybrid"
    NEURAL_LINEAR = "neural_linear"
    THOMPSON = "thompson"
    UCB1 = "ucb1"
    EPSILON_GREEDY = "epsilon_greedy"
    BOOTSTRAPPED_TS = "bootstrapped_ts"
    BOOTSTRAPPED_UCB = "bootstrapped_ucb"
    LOGISTIC_UCB = "logistic_ucb"
    LOGISTIC_TS = "logistic_ts"
    GP_UCB = "gp_ucb"
    SOFTMAX = "softmax"
    LIN_UCB_SW = "linucb_sw"  # Sliding-window LinUCB
    CATS = "cats"  # Continuous Action Tree Sampling
