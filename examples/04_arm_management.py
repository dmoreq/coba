"""
04_arm_management.py — Dynamic arm lifecycle: add, warm-start, and remove arms.

Scenario: a product catalogue bandit starts with three price tiers,
then adds a new "premium" tier initialised from the best existing arm,
and later retires an underperforming arm.
"""

import numpy as np
from coba import ClusterBandit, PolicyType

rng = np.random.default_rng(3)
N_FEATURES = 4
ARMS = ["budget", "standard", "deluxe"]

bandit = ClusterBandit(arms=ARMS, n_features=N_FEATURES, policy=PolicyType.LIN_UCB, seed=0)

# Fit on initial logs
contexts = rng.standard_normal((300, N_FEATURES))
decisions = rng.choice(ARMS, size=300)
rewards = np.clip(rng.normal(0.5, 0.2, size=300), 0, 1)
bandit.fit_offline(contexts, decisions, rewards)

test_ctx = rng.standard_normal(N_FEATURES)
print("Arms before:", bandit.arms)
print("Decision before:", bandit.decide(test_ctx).chosen_arm)

# --- Add a new arm, warm-started from the best current arm ---
# "premium" starts with the "deluxe" arm's learned model to avoid cold start
bandit.add_arm("premium", warm_start_from="deluxe")
print("\nArms after add:", bandit.arms)
print("Decision after add:", bandit.decide(test_ctx).chosen_arm)

# Update the new arm with a few online observations
for _ in range(20):
    ctx = rng.standard_normal(N_FEATURES)
    reward = float(np.clip(rng.normal(0.75, 0.1), 0, 1))
    bandit.update(ctx, "premium", reward)

print("Decision after premium updates:", bandit.decide(test_ctx).chosen_arm)

# --- Remove an underperforming arm ---
bandit.remove_arm("budget")
print("\nArms after remove:", bandit.arms)
print("Decision after remove:", bandit.decide(test_ctx).chosen_arm)
