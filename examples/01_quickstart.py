"""
01_quickstart.py — Minimal create → decide → update loop.

Scenario: choose a discount multiplier (arm) for each incoming user,
observe a conversion reward, and update the bandit.
"""

import numpy as np
from coba import ClusterBandit, PolicyType

rng = np.random.default_rng(0)

# Arms represent discount multipliers (1.0x = no discount, 1.5x = 50% off)
bandit = ClusterBandit(
    arms=[1.0, 1.1, 1.2, 1.5],
    n_features=4,
    policy=PolicyType.LIN_UCB,
    n_clusters=3,
    seed=42,
)

# Simulate an initial batch of logged data so the bandit is fitted
n_init = 200
contexts = rng.standard_normal((n_init, 4))
decisions = rng.choice([1.0, 1.1, 1.2, 1.5], size=n_init)
rewards = np.clip(rng.normal(0.5, 0.2, size=n_init), 0, 1)

bandit.fit_offline(contexts, decisions, rewards)
print("Bandit fitted:", bandit.is_fitted)

# --- Online serving loop ---
for step in range(10):
    user_context = rng.standard_normal(4)

    decision = bandit.decide(user_context)
    print(f"Step {step+1}: chose arm={decision.chosen_arm}, score={decision.score:.3f}")

    # Simulate reward (higher discount → slightly higher conversion)
    base_reward = 0.3 + 0.1 * float(decision.chosen_arm)
    observed_reward = float(np.clip(rng.normal(base_reward, 0.1), 0, 1))

    bandit.update(
        context=user_context,
        arm=decision.chosen_arm,
        reward=observed_reward,
    )
