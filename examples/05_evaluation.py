"""
05_evaluation.py — Offline policy evaluation: rejection sampling, DR, and NCIS.

All three estimators measure the expected reward of the *learned* policy
using only the historical log — no live traffic needed.
"""

import numpy as np
from sklearn.linear_model import Ridge

from coba import ClusterBandit, PolicyType

rng = np.random.default_rng(4)
N_FEATURES = 5
ARMS = [1.0, 1.2, 1.5]
N_LOGS = 1000

# --- Generate logged data with a uniform logging policy ---
contexts = rng.standard_normal((N_LOGS, N_FEATURES))
decisions = rng.choice(ARMS, size=N_LOGS)
propensities = np.full(N_LOGS, 1.0 / len(ARMS))
true_means = {1.0: 0.4, 1.2: 0.6, 1.5: 0.5}
rewards = np.array([float(np.clip(rng.normal(true_means[a], 0.15), 0, 1)) for a in decisions])

# Fit bandit on the logs
bandit = ClusterBandit(arms=ARMS, n_features=N_FEATURES, policy=PolicyType.LIN_UCB, seed=0)
bandit.fit_offline(contexts, decisions, rewards, propensities=propensities)

# --- 1. Rejection Sampling ---
# Unbiased only for uniform logging; wastes samples where log ≠ target action
rs_result = bandit.evaluate_rejection_sampling(contexts, decisions, rewards)
print(rs_result)

# --- 2. Doubly-Robust ---
# Needs reward model estimates for the logged arm
reward_model = Ridge(alpha=1.0).fit(contexts, rewards)
reward_estimates = reward_model.predict(contexts)

# Also compute estimates for the target policy's action
target_decisions = np.array([bandit.decide(ctx).chosen_arm for ctx in contexts])
target_reward_estimates = reward_model.predict(contexts)  # same model, different actions

dr_result = bandit.evaluate_doubly_robust(
    contexts,
    decisions,
    rewards,
    propensities,
    reward_estimates,
    target_reward_estimates=target_reward_estimates,
)
print(dr_result)

# --- 3. NCIS (Normalized Capped Importance Sampling) ---
# Use policy scores vs. logging scores as importance weights
policy_scores = np.array(
    [bandit.score_all(ctx).get(a, 1e-6) for ctx, a in zip(contexts, decisions)]
)
logging_scores = propensities  # uniform logging → constant score

ncis_result = bandit.evaluate_ncis(
    policy_scores=policy_scores,
    logging_scores=logging_scores,
    rewards=rewards,
)
print(ncis_result)
