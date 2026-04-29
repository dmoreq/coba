"""
06_persistence.py — Save and reload a trained bandit with save_bandit/load_bandit.

The loaded bandit produces identical decisions to the original, with no
re-training needed — useful for deploying a bandit to a serving fleet.
"""

import tempfile
from pathlib import Path

import numpy as np
from coba import ClusterBandit, PolicyType
from coba.persistence import load_bandit, save_bandit

rng = np.random.default_rng(5)
N_FEATURES = 4
ARMS = ["A", "B", "C"]

# Train a bandit
bandit = ClusterBandit(arms=ARMS, n_features=N_FEATURES, policy=PolicyType.LIN_UCB, seed=0)
contexts = rng.standard_normal((200, N_FEATURES))
decisions = rng.choice(ARMS, size=200)
rewards = np.clip(rng.normal(0.5, 0.2, size=200), 0, 1)
bandit.fit_offline(contexts, decisions, rewards)

test_ctx = rng.standard_normal(N_FEATURES)
original_decision = bandit.decide(test_ctx)
print(f"Original  → arm={original_decision.chosen_arm}, score={original_decision.score:.4f}")

# Save to disk
with tempfile.TemporaryDirectory() as tmpdir:
    model_path = Path(tmpdir) / "bandit.joblib"
    save_bandit(bandit, model_path)
    print(f"Saved to: {model_path}")

    # Load from disk
    loaded_bandit = load_bandit(model_path)
    loaded_decision = loaded_bandit.decide(test_ctx)
    print(f"Reloaded  → arm={loaded_decision.chosen_arm}, score={loaded_decision.score:.4f}")

assert original_decision.chosen_arm == loaded_decision.chosen_arm, "Decision mismatch after reload!"
print("Round-trip check passed.")
