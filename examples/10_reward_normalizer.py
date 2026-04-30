"""
10_reward_normalizer.py — Scale raw business metrics before feeding the bandit.

Scenario: optimizing an e-commerce discount. Rewards are raw order values
(€50–€500). Thompson Sampling requires rewards in [0, 1]. RewardNormalizer
handles the conversion using running min-max statistics so no pre-computation
of reward bounds is needed.
"""

import numpy as np

from coba import ClusterBandit, RewardNormalizer
from coba.types import PolicyType

rng = np.random.default_rng(0)

bandit = ClusterBandit(
    arms=["5%", "10%", "15%", "20%"],
    n_features=3,
    policy=PolicyType.THOMPSON,  # requires rewards in [0, 1]
    n_clusters=1,
    seed=0,
)

# Raw order values — not in [0, 1]
normalizer = RewardNormalizer(mode="minmax", decay=0.999)

print("Step | Arm  | Raw reward | Normalized | is_fitted")
print("-----|------|------------|------------|----------")

for step in range(1, 21):
    ctx = rng.standard_normal(3)
    decision = bandit.decide(ctx)
    arm = decision.chosen_arm

    # Simulate raw business reward (order value in euros)
    raw_reward = rng.uniform(50.0, 500.0)

    # Normalize before updating the bandit
    normed_reward = normalizer.update_and_normalize(raw_reward)
    bandit.update(context=ctx, arm=arm, reward=normed_reward)

    print(
        f" {step:3d} | {arm:<4} | {raw_reward:10.2f} | {normed_reward:10.4f} | {normalizer.is_fitted}"
    )

# --- zscore mode for linear policies ---
print("\nz-score mode (suitable for LinUCB/LinTS with unbounded rewards):")
zscore_norm = RewardNormalizer(mode="zscore", decay=0.99)
for raw in [100.0, 200.0, 150.0, 300.0, 50.0]:
    normed = zscore_norm.update_and_normalize(raw)
    print(f"  raw={raw:.1f}  zscore={normed:+.4f}")

# --- normalize() vs update_and_normalize() ---
print("\nnormalize() (read-only) vs update_and_normalize() (updates stats):")
norm = RewardNormalizer(mode="minmax")
norm.update_and_normalize(0.0)
norm.update_and_normalize(100.0)
print(f"  normalize(50)  = {norm.normalize(50.0):.4f}  (stats unchanged)")
print(f"  update(50)     = {norm.update_and_normalize(50.0):.4f}  (stats updated)")
