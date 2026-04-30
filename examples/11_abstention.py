"""
11_abstention.py — Confidence-based abstention with min_confidence_gap.

Scenario: a pricing bandit should only act when it's confident. When the
top two arms have nearly identical scores, it defers to a rule-based fallback
rather than making an arbitrary choice.
"""

import numpy as np

from coba import ClusterBandit
from coba.types import PolicyType

rng = np.random.default_rng(0)

bandit = ClusterBandit(
    arms=["price_A", "price_B", "price_C"],
    n_features=4,
    policy=PolicyType.LIN_UCB,
    n_clusters=2,
    seed=0,
)

n = 600
contexts = rng.standard_normal((n, 4))
decisions = rng.choice(bandit.arms, size=n)
rewards = rng.uniform(0, 1, n)
bandit.fit_offline(contexts, decisions, rewards)

FALLBACK_ARM = "price_A"

n_decided = n_abstained = 0
for _ in range(50):
    ctx = rng.standard_normal(4)
    decision = bandit.decide(ctx, min_confidence_gap=0.05)

    if decision.abstained:
        chosen = FALLBACK_ARM
        n_abstained += 1
    else:
        chosen = decision.chosen_arm  # type: ignore[assignment]
        n_decided += 1

    bandit.update(context=ctx, arm=chosen, reward=rng.uniform(0, 1))

print(f"Decided:  {n_decided}")
print(f"Abstained (fell back to '{FALLBACK_ARM}'): {n_abstained}")

# --- Check abstained decision structure ---
high_gap_decision = bandit.decide(rng.standard_normal(4), min_confidence_gap=1e9)
print(
    f"\nWith gap=1e9: abstained={high_gap_decision.abstained}, chosen_arm={high_gap_decision.chosen_arm}"
)
print(f"All scores still present: {list(high_gap_decision.all_scores.keys())}")
