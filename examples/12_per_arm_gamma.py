"""
12_per_arm_gamma.py — Per-arm decay rates for dynamic arm management.

Scenario: an existing bandit has stable, well-calibrated arms. A new arm is
introduced mid-flight (e.g. a new product variant). The new arm should adapt
quickly to its own reward distribution while the old arms stay stable.

Per-arm gamma lets you give the new arm a high decay (fast adaptation)
without destabilising the existing models.
"""

import numpy as np

from coba import ClusterBandit
from coba.types import PolicyType

rng = np.random.default_rng(0)

bandit = ClusterBandit(
    arms=["control", "variant_A", "variant_B"],
    n_features=3,
    policy=PolicyType.LIN_UCB,
    n_clusters=2,
    gamma=1.0,  # global: no decay (stable, long-running arms)
    seed=0,
)

# Bootstrap existing arms on historical data
n = 500
contexts = rng.standard_normal((n, 3))
decisions = rng.choice(bandit.arms, size=n)
rewards = rng.uniform(0, 1, n)
bandit.fit_offline(contexts, decisions, rewards)

print("Arms after bootstrap:", bandit.arms)

# Launch a new arm mid-flight with faster adaptation (gamma=0.95)
bandit.add_arm("variant_C", gamma=0.95)
print(f"After add_arm('variant_C', gamma=0.95): {bandit.arms}")

# Simulate ongoing traffic — new arm quickly learns its reward pattern
for step in range(30):
    ctx = rng.standard_normal(3)
    decision = bandit.decide(ctx)
    arm = decision.chosen_arm
    reward = rng.uniform(0, 1)
    bandit.update(context=ctx, arm=arm, reward=reward)

# New arm should be reachable in decisions
chosen_arms = set()
for _ in range(100):
    ctx = rng.standard_normal(3)
    decision = bandit.decide(ctx)
    if decision.chosen_arm:
        chosen_arms.add(decision.chosen_arm)

print(f"Arms chosen in 100 decisions: {chosen_arms}")
print("variant_C explored:", "variant_C" in chosen_arms)

# --- Compare: adding without gamma uses global gamma=1.0 ---
bandit.add_arm("variant_D")  # gamma defaults to bandit's global gamma (1.0)
print(f"\nAfter add_arm('variant_D') with default gamma=1.0: {bandit.arms}")
