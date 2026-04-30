"""
15_lin_ucb_hybrid.py — LinUCB-Hybrid: shared + per-arm feature learning.

Reference: Li et al., "A contextual-bandit approach to personalized news", WWW 2010.

Scenario: a news recommendation system with two kinds of features:
  - Shared features (z): user demographics, time-of-day, device type (3 dims)
    These are the same regardless of which article is shown.
  - Arm-specific features (x): article embedding, topic vector (4 dims)
    These vary per article.

With regular LinUCB each arm learns user features independently from its own
pulls. With LinUCB-Hybrid ALL arms contribute to learning the shared user
model (β_shared), so a data-poor arm still benefits from what other arms
learned about user preferences.

Context layout: [z_0, z_1, z_2, x_0, x_1, x_2, x_3]
                 ←── shared (3) ──→ ←──── per-arm (4) ────→
"""

import numpy as np

from coba import ClusterBandit
from coba.types import PolicyType

rng = np.random.default_rng(42)

ARMS = ["article_A", "article_B", "article_C", "article_D"]
N_SHARED = 3  # user/session features (shared across arms)
N_ARM = 4  # article-specific features
N_FEATURES = N_SHARED + N_ARM

# True reward weights (unknown to the bandit):
#   shared part: same weight for all arms (captures user preference)
#   arm part: different per arm (captures article quality for that user type)
W_SHARED = np.array([0.3, -0.2, 0.4])
W_ARM = {
    "article_A": np.array([0.5, 0.1, -0.3, 0.2]),
    "article_B": np.array([-0.2, 0.4, 0.1, 0.3]),
    "article_C": np.array([0.1, -0.1, 0.5, -0.2]),
    "article_D": np.array([0.3, 0.2, 0.0, 0.4]),
}


def true_reward(arm: str, ctx: np.ndarray) -> float:
    z, x_arm = ctx[:N_SHARED], ctx[N_SHARED:]
    r = float(z @ W_SHARED + x_arm @ W_ARM[arm])
    return float(np.clip(r + rng.normal(0, 0.1), 0, 1))


# --------------------------------------------------------------------------
# Build hybrid bandit
# --------------------------------------------------------------------------
bandit = ClusterBandit(
    arms=ARMS,
    n_features=N_FEATURES,
    policy=PolicyType.LIN_UCB_HYBRID,
    n_shared_features=N_SHARED,  # first 3 dims are shared
    n_clusters=2,
    alpha=0.5,
    seed=0,
)

# Bootstrap on historical logs
n_boot = 600
boot_ctx = rng.standard_normal((n_boot, N_FEATURES))
boot_arms = rng.choice(ARMS, size=n_boot)
boot_rewards = np.array([true_reward(a, c) for a, c in zip(boot_arms, boot_ctx)])
bandit.fit_offline(boot_ctx, boot_arms, boot_rewards)

# --------------------------------------------------------------------------
# Online simulation
# --------------------------------------------------------------------------
total_reward = 0.0
for step in range(500):
    ctx = rng.standard_normal(N_FEATURES)
    decision = bandit.decide(ctx)
    arm = decision.chosen_arm
    reward = true_reward(arm, ctx)
    bandit.update(context=ctx, arm=arm, reward=reward)
    total_reward += reward

mean_reward = total_reward / 500
print(f"Mean reward over 500 steps: {mean_reward:.4f}")

# --------------------------------------------------------------------------
# Compare with standard LinUCB (disjoint — no shared features)
# --------------------------------------------------------------------------
bandit_disjoint = ClusterBandit(
    arms=ARMS,
    n_features=N_FEATURES,
    policy=PolicyType.LIN_UCB,
    n_clusters=2,
    alpha=0.5,
    seed=0,
)
bandit_disjoint.fit_offline(boot_ctx, boot_arms, boot_rewards)

total_disjoint = 0.0
rng2 = np.random.default_rng(42)  # same seed for fair comparison
for step in range(500):
    ctx = rng2.standard_normal(N_FEATURES)
    decision = bandit_disjoint.decide(ctx)
    arm = decision.chosen_arm
    reward = true_reward(arm, ctx)
    bandit_disjoint.update(context=ctx, arm=arm, reward=reward)
    total_disjoint += reward

mean_disjoint = total_disjoint / 500
print(f"Mean reward (standard LinUCB, disjoint): {mean_disjoint:.4f}")
print(f"\nHybrid improvement: {(mean_reward - mean_disjoint):.4f}")
print("(Hybrid benefits from cross-arm shared feature learning)")
