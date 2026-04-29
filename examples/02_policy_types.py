"""
02_policy_types.py — Instantiate and exercise all PolicyType variants.

Each policy is fitted on the same synthetic dataset so the output is
directly comparable.
"""

import numpy as np
from sklearn.linear_model import SGDRegressor

from coba import ClusterBandit, PolicyType

rng = np.random.default_rng(1)
N_FEATURES = 5
ARMS = ["a", "b", "c"]
N_INIT = 300

# Shared synthetic log
contexts = rng.standard_normal((N_INIT, N_FEATURES))
decisions = rng.choice(ARMS, size=N_INIT)
rewards = np.clip(rng.normal(0.5, 0.2, size=N_INIT), 0, 1)
test_ctx = rng.standard_normal(N_FEATURES)

# Policies that need no extra args
simple_policies = [
    PolicyType.LIN_UCB,
    PolicyType.LIN_TS,
    PolicyType.LOGISTIC_UCB,
    PolicyType.LOGISTIC_TS,
    PolicyType.THOMPSON,
    PolicyType.UCB1,
]

for policy in simple_policies:
    bandit = ClusterBandit(arms=ARMS, n_features=N_FEATURES, policy=policy, n_clusters=3, seed=0)
    bandit.fit_offline(contexts, decisions, rewards)
    decision = bandit.decide(test_ctx)
    print(f"{policy.value:20s}  →  arm={decision.chosen_arm}, score={decision.score:.4f}")

print()

# Sklearn-backed policies require a base_estimator
sklearn_policies = [
    (PolicyType.EPSILON_GREEDY, {"epsilon": 0.1}),
    (PolicyType.BOOTSTRAPPED_TS, {"n_bootstraps": 8}),
    (PolicyType.BOOTSTRAPPED_UCB, {"n_bootstraps": 8}),
]

for policy, extra in sklearn_policies:
    bandit = ClusterBandit(
        arms=ARMS,
        n_features=N_FEATURES,
        policy=policy,
        n_clusters=3,
        seed=0,
        base_estimator=SGDRegressor(max_iter=1, tol=None, random_state=0),
        **extra,
    )
    bandit.fit_offline(contexts, decisions, rewards)
    decision = bandit.decide(test_ctx)
    print(f"{policy.value:20s}  →  arm={decision.chosen_arm}, score={decision.score:.4f}")
