"""
03_offline_bootstrap.py — Bootstrap from historical logs with IPS and DR correction.

Scenario: a rule-based logging policy favoured arm "high" heavily.
We correct for this bias before handing control to the learned bandit.
"""

import numpy as np
from sklearn.linear_model import Ridge

from coba import ClusterBandit, PolicyType

rng = np.random.default_rng(2)
N_FEATURES = 6
ARMS = ["low", "medium", "high"]
N_LOGS = 500

# Simulate biased logs: logging policy chose "high" 70% of the time
logging_probs = {"low": 0.10, "medium": 0.20, "high": 0.70}
contexts = rng.standard_normal((N_LOGS, N_FEATURES))
decisions = rng.choice(ARMS, p=list(logging_probs.values()), size=N_LOGS)
propensities = np.array([logging_probs[a] for a in decisions])
# True reward is highest for "low" arm (the logging policy was suboptimal)
true_means = {"low": 0.8, "medium": 0.5, "high": 0.3}
rewards = np.array([float(np.clip(rng.normal(true_means[a], 0.1), 0, 1)) for a in decisions])

# -------------------------------------------------------------------
# Pure IPS bootstrap
# -------------------------------------------------------------------
bandit_ips = ClusterBandit(arms=ARMS, n_features=N_FEATURES, policy=PolicyType.LIN_UCB, seed=0)
bandit_ips.fit_offline(contexts, decisions, rewards, propensities=propensities)

test_ctx = rng.standard_normal(N_FEATURES)
dec_ips = bandit_ips.decide(test_ctx)
print(f"IPS bootstrap  → chosen arm: {dec_ips.chosen_arm}")

# -------------------------------------------------------------------
# Doubly-Robust bootstrap (requires reward model estimates)
# -------------------------------------------------------------------
# Fit a simple reward model on the logged data
reward_model = Ridge(alpha=1.0)
reward_model.fit(contexts, rewards)
reward_estimates = reward_model.predict(contexts)

bandit_dr = ClusterBandit(arms=ARMS, n_features=N_FEATURES, policy=PolicyType.LIN_UCB, seed=0)
bandit_dr.fit_offline(
    contexts,
    decisions,
    rewards,
    propensities=propensities,
    use_dr=True,
    reward_estimates=reward_estimates,
)

dec_dr = bandit_dr.decide(test_ctx)
print(f"DR  bootstrap  → chosen arm: {dec_dr.chosen_arm}")
print("(Expected: 'low' — true best arm hidden from the biased logging policy)")
