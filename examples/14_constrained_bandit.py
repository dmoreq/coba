"""
14_constrained_bandit.py — Minimum pull-rate constraints for guaranteed exploration.

Scenario: a content recommendation bandit must satisfy editorial requirements:
  - "sponsored" slot must appear in at least 10% of decisions (contractual)
  - "new_feature" must get at least 5% traffic for an A/B test to be valid

Without constraints the bandit may never pull low-performing arms once it has
enough data, making it impossible to detect improvements. min_pull_rates forces
the constraint by intercepting the scoring step: any arm below its target rate
is injected as the only candidate until the constraint is satisfied.
"""

import numpy as np

from coba import ClusterBandit
from coba.types import PolicyType

rng = np.random.default_rng(0)

ARMS = ["hero_banner", "sponsored", "editorial", "new_feature"]
N_FEATURES = 4

bandit = ClusterBandit(
    arms=ARMS,
    n_features=N_FEATURES,
    policy=PolicyType.LIN_UCB,
    n_clusters=2,
    seed=0,
    min_pull_rates={
        "sponsored": 0.10,  # contractual floor
        "new_feature": 0.05,  # A/B test floor
    },
)

# Bootstrap on historical data
bandit.fit_offline(
    contexts=rng.standard_normal((400, N_FEATURES)),
    decisions=rng.choice(ARMS, size=400),
    rewards=rng.uniform(0, 1, 400),
)

# Simulate 500 live decisions
pull_counts = {arm: 0 for arm in ARMS}
for _ in range(500):
    ctx = rng.standard_normal(N_FEATURES)
    decision = bandit.decide(ctx)
    arm = decision.chosen_arm
    pull_counts[arm] += 1
    # Simulate rewards: hero and editorial convert well; sponsored and new_feature less so
    if arm in ("hero_banner", "editorial"):
        reward = rng.normal(0.65, 0.1)
    else:
        reward = rng.normal(0.35, 0.1)
    bandit.update(context=ctx, arm=arm, reward=float(np.clip(reward, 0, 1)))

total = sum(pull_counts.values())
print("Pull rates after 500 decisions:")
for arm in ARMS:
    rate = pull_counts[arm] / total
    flag = (
        " ✓"
        if arm not in ("sponsored", "new_feature")
        else (" ✓" if rate >= 0.10 - 0.01 else " ✗ (below floor)")
    )
    print(f"  {arm:<16} {rate:.1%}{flag}")

# Verify constraints are satisfied
sponsored_rate = pull_counts["sponsored"] / total
new_feature_rate = pull_counts["new_feature"] / total
assert sponsored_rate >= 0.09, f"sponsored rate {sponsored_rate:.2%} below 10% floor"
assert new_feature_rate >= 0.04, f"new_feature rate {new_feature_rate:.2%} below 5% floor"
print("\nAll minimum pull-rate constraints satisfied.")

# --- Show that removing an arm also removes its constraint ---
bandit.remove_arm("sponsored")
print(f"\nAfter removing 'sponsored': constrained arms = {list(bandit._min_pull_rates or {})}")
