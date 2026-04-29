"""
07_batch_update.py — Batch and streaming log-replay with update_batch.

Two patterns:
  A) One-shot batch update from a historical log file snapshot.
  B) Streaming ingestion: process log chunks as they arrive.

update_batch() uses existing cluster centroids (no KMeans refit),
so cluster assignments remain stable across updates.
"""

import numpy as np
from coba import ClusterBandit, PolicyType

rng = np.random.default_rng(6)
N_FEATURES = 5
ARMS = [0.5, 1.0, 2.0]
N_INIT = 300

# Bootstrap the bandit first so clusters are fitted
bandit = ClusterBandit(arms=ARMS, n_features=N_FEATURES, policy=PolicyType.LIN_UCB, seed=0)
init_ctx = rng.standard_normal((N_INIT, N_FEATURES))
init_dec = rng.choice(ARMS, size=N_INIT)
init_rew = np.clip(rng.normal(0.5, 0.2, size=N_INIT), 0, 1)
bandit.fit_offline(init_ctx, init_dec, init_rew)

# ----------------------------------------------------------------
# Pattern A: one-shot batch update from a daily log snapshot
# ----------------------------------------------------------------
batch_ctx = rng.standard_normal((500, N_FEATURES))
batch_dec = rng.choice(ARMS, size=500)
batch_rew = np.clip(rng.normal(0.6, 0.15, size=500), 0, 1)
batch_prop = np.full(500, 1.0 / len(ARMS))  # uniform logging policy

bandit.update_batch(batch_ctx, batch_dec, batch_rew, propensities=batch_prop)
print("After one-shot batch update:")
print("  Stats:", [(s.arm, s.n_pulls) for s in bandit.get_stats()])

# ----------------------------------------------------------------
# Pattern B: streaming ingestion — process chunks of 100 rows
# ----------------------------------------------------------------
CHUNK_SIZE = 100
N_CHUNKS = 5

for chunk_idx in range(N_CHUNKS):
    chunk_ctx = rng.standard_normal((CHUNK_SIZE, N_FEATURES))
    chunk_dec = rng.choice(ARMS, size=CHUNK_SIZE)
    chunk_rew = np.clip(rng.normal(0.55, 0.15, size=CHUNK_SIZE), 0, 1)

    bandit.update_batch(chunk_ctx, chunk_dec, chunk_rew)
    total_pulls = sum(s.n_pulls for s in bandit.get_stats())
    print(f"  Chunk {chunk_idx+1}: cumulative pulls={total_pulls}")

test_ctx = rng.standard_normal(N_FEATURES)
print("Final decision:", bandit.decide(test_ctx).chosen_arm)
