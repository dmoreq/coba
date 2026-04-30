"""
09_top_k_decisions.py — Ranked arm selection with decide_top_k.

Scenario: a recommendation system needs to show a ranked list of 3 offers
instead of just one. decide_top_k() returns arms ordered by score so the
caller can display the best option first and fall back to alternatives.
"""

import numpy as np

from coba import ClusterBandit
from coba.types import PolicyType

rng = np.random.default_rng(0)

bandit = ClusterBandit(
    arms=["premium", "standard", "basic", "trial", "free"],
    n_features=4,
    policy=PolicyType.LIN_TS,
    n_clusters=3,
    seed=0,
)

# Bootstrap from historical logs
n = 800
contexts = rng.standard_normal((n, 4))
decisions = rng.choice(bandit.arms, size=n)
rewards = rng.uniform(0, 1, n)
bandit.fit_offline(contexts, decisions, rewards)

# --- Single-best decision (standard) ---
ctx = rng.standard_normal(4)
decision = bandit.decide(ctx)
print(f"Best arm:  {decision.chosen_arm}  (score={decision.score:.4f})")

# --- Ranked top-3 (new) ---
top3 = bandit.decide_top_k(ctx, k=3)
print("\nTop-3 ranked arms:")
for rank, (arm, score) in enumerate(top3, start=1):
    print(f"  #{rank}: {arm:<10}  score={score:.4f}")

# --- Practical use: show top-K, update only on the arm the user interacted with ---
shown_arms = [arm for arm, _ in bandit.decide_top_k(ctx, k=3)]
user_clicked_rank = 2  # user clicked the second offer
chosen = shown_arms[user_clicked_rank - 1]
bandit.update(context=ctx, arm=chosen, reward=1.0)
print(f"\nUser clicked rank #{user_clicked_rank}: '{chosen}' — bandit updated.")

# --- k > n_arms is safe (clamped silently) ---
all_ranked = bandit.decide_top_k(ctx, k=100)
print(f"\nRequested k=100, got {len(all_ranked)} arms (clamped to arm count).")
