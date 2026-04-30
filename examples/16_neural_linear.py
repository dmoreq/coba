"""
16_neural_linear.py — NeuralLinear bandit: non-linear feature extraction + LinTS.

Reference: Riquelme et al., "Deep Bayesian Bandits Showdown", ICLR 2018.

Architecture:
  - A shared MLP backbone (sklearn MLPRegressor) per cluster extracts a
    low-dimensional embedding from the raw context.
  - Each arm maintains a per-arm LinTS on top of the embedding.
  - The backbone retrains periodically on a replay buffer of all observations.
  - Between retrains, LinTS updates online using the current embedding.

When to use NEURAL_LINEAR vs. LIN_UCB:
  Use NeuralLinear when the true reward function is non-linear in the raw
  context features. The MLP backbone learns a manifold where the reward becomes
  approximately linear, then LinTS does efficient Bayesian uncertainty estimation.

Scenario: a recommendation system where the reward depends on a non-linear
  interaction between user features and item features (e.g. their dot product
  or a threshold effect that a linear model cannot capture).
"""

import numpy as np

from coba import ClusterBandit
from coba.types import PolicyType

rng = np.random.default_rng(42)

ARMS = ["item_A", "item_B", "item_C", "item_D"]
N_FEATURES = 8

# Non-linear reward: sigmoid of a quadratic interaction
_ARM_WEIGHTS = {
    "item_A": np.array([1.0, -0.5, 0.3, 0.8, -0.2, 0.1, 0.5, -0.1]),
    "item_B": np.array([-0.3, 0.9, 0.4, -0.2, 0.7, -0.3, 0.1, 0.6]),
    "item_C": np.array([0.5, 0.2, -0.8, 0.4, 0.1, 0.9, -0.4, 0.2]),
    "item_D": np.array([-0.1, -0.4, 0.7, 0.3, -0.6, 0.2, 0.8, -0.3]),
}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def true_reward(arm: str, ctx: np.ndarray) -> float:
    w = _ARM_WEIGHTS[arm]
    # Non-linear: sigmoid of (x @ w) + 0.3 * (x @ w)^2
    linear = float(ctx @ w)
    reward = sigmoid(linear + 0.3 * linear**2)
    return float(np.clip(reward + rng.normal(0, 0.05), 0, 1))


# --------------------------------------------------------------------------
# NeuralLinear bandit
# --------------------------------------------------------------------------
bandit = ClusterBandit(
    arms=ARMS,
    n_features=N_FEATURES,
    policy=PolicyType.NEURAL_LINEAR,
    n_clusters=2,
    neural_embedding_dim=16,  # penultimate layer size (LinTS input)
    neural_hidden_sizes=(32,),  # hidden layers before embedding
    neural_retrain_freq=100,  # retrain backbone every 100 total updates
    seed=0,
)

# Bootstrap on historical data
n_boot = 400
boot_ctx = rng.standard_normal((n_boot, N_FEATURES))
boot_arms = rng.choice(ARMS, size=n_boot)
boot_rewards = np.array([true_reward(a, c) for a, c in zip(boot_arms, boot_ctx)])
bandit.fit_offline(boot_ctx, boot_arms, boot_rewards)

print("Starting online simulation (500 steps)...")

# Online loop
total_reward_nl = 0.0
for step in range(500):
    ctx = rng.standard_normal(N_FEATURES)
    decision = bandit.decide(ctx)
    arm = decision.chosen_arm
    reward = true_reward(arm, ctx)
    bandit.update(context=ctx, arm=arm, reward=reward)
    total_reward_nl += reward

mean_nl = total_reward_nl / 500
print(f"NeuralLinear mean reward:   {mean_nl:.4f}")

# --------------------------------------------------------------------------
# Baseline: standard LinUCB (linear model, no feature extraction)
# --------------------------------------------------------------------------
bandit_lin = ClusterBandit(
    arms=ARMS, n_features=N_FEATURES, policy=PolicyType.LIN_UCB, n_clusters=2, seed=0
)
rng2 = np.random.default_rng(42)
bandit_lin.fit_offline(boot_ctx, boot_arms, boot_rewards)

total_reward_lin = 0.0
for step in range(500):
    ctx = rng2.standard_normal(N_FEATURES)
    decision = bandit_lin.decide(ctx)
    arm = decision.chosen_arm
    reward = true_reward(arm, ctx)
    bandit_lin.update(context=ctx, arm=arm, reward=reward)
    total_reward_lin += reward

mean_lin = total_reward_lin / 500
print(f"LinUCB mean reward:         {mean_lin:.4f}")
print(f"\nNeuralLinear vs LinUCB:     {mean_nl - mean_lin:+.4f}")
print("Note: NeuralLinear needs more data and retrain cycles to outperform LinUCB.")
print("It shines on strongly non-linear reward functions with large datasets.")
