"""
08_monitoring.py — Inspect bandit internals for dashboards and debugging.

Covers:
  - get_stats()              per-arm pull counts and mean reward
  - score_all()              raw per-arm scores for a single context
  - get_cluster_assignment() which KMeans cluster a context falls into
  - BanditStats fields       arm, n_pulls, mean_reward
"""

import numpy as np
from coba import ClusterBandit, PolicyType

rng = np.random.default_rng(7)
N_FEATURES = 4
ARMS = ["cheap", "mid", "premium"]
N_CLUSTERS = 4

bandit = ClusterBandit(
    arms=ARMS,
    n_features=N_FEATURES,
    policy=PolicyType.LIN_UCB,
    n_clusters=N_CLUSTERS,
    seed=0,
)

# Fit and run a short online loop so we have non-trivial stats
contexts = rng.standard_normal((300, N_FEATURES))
decisions = rng.choice(ARMS, size=300)
rewards = np.clip(rng.normal(0.5, 0.2, size=300), 0, 1)
bandit.fit_offline(contexts, decisions, rewards)

for _ in range(50):
    ctx = rng.standard_normal(N_FEATURES)
    dec = bandit.decide(ctx)
    reward = float(np.clip(rng.normal(0.5, 0.15), 0, 1))
    bandit.update(ctx, dec.chosen_arm, reward)

# --- Per-arm statistics ---
print("=== Per-arm stats ===")
for stats in bandit.get_stats():
    print(f"  arm={stats.arm!r:10s}  pulls={stats.n_pulls:4d}  mean_reward={stats.mean_reward:.3f}")

# --- Raw scores for a specific context ---
test_ctx = rng.standard_normal(N_FEATURES)
print("\n=== Scores for test context ===")
scores = bandit.score_all(test_ctx)
for arm, score in scores.items():
    print(f"  arm={arm!r:10s}  score={score:.4f}")

# --- Cluster assignment ---
cluster_idx = bandit.get_cluster_assignment(test_ctx)
print("\n=== Cluster assignment ===")
print(f"  test_ctx falls into cluster {cluster_idx} (of {bandit.n_clusters})")

# --- Cluster assignment distribution across many contexts ---
sample_contexts = rng.standard_normal((200, N_FEATURES))
cluster_counts = {}
for ctx in sample_contexts:
    c = bandit.get_cluster_assignment(ctx)
    cluster_counts[c] = cluster_counts.get(c, 0) + 1

print("\n=== Cluster distribution (200 random contexts) ===")
for cluster, count in sorted(cluster_counts.items()):
    bar = "#" * (count // 4)
    print(f"  cluster {cluster}: {count:3d} {bar}")
