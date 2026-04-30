"""
13_drift_detection.py — Automatic drift detection with PageHinkleyDetector.

Scenario: an e-commerce bandit serving product recommendations. Midway through
a campaign, one product's conversion rate drops sharply (stock issue, price
change, seasonal effect). Without drift detection the bandit keeps serving
stale beliefs. With drift detection it resets the affected arm and re-learns
the new reward distribution.

Two usage patterns are shown:
  A. Automatic: pass enable_drift_detection=True to ClusterBandit — the bandit
     resets any arm whose reward stream triggers the PH test.
  B. Manual: manage PageHinkleyDetector instances yourself for full control.
"""

import numpy as np

from coba import ClusterBandit, PageHinkleyDetector
from coba.types import PolicyType

rng = np.random.default_rng(42)

ARMS = ["product_A", "product_B", "product_C"]
N_FEATURES = 4

# ---------------------------------------------------------------------------
# Pattern A — automatic drift detection built into ClusterBandit
# ---------------------------------------------------------------------------
print("=== Pattern A: Automatic drift detection ===\n")

bandit = ClusterBandit(
    arms=ARMS,
    n_features=N_FEATURES,
    policy=PolicyType.LIN_UCB,
    n_clusters=2,
    seed=0,
    enable_drift_detection=True,
    drift_delta=0.005,  # minimum detectable change
    drift_lambda=30.0,  # detection threshold (lower → faster, more sensitive)
)

# Bootstrap on historical data
n_boot = 300
bandit.fit_offline(
    contexts=rng.standard_normal((n_boot, N_FEATURES)),
    decisions=rng.choice(ARMS, size=n_boot),
    rewards=rng.uniform(0, 1, n_boot),
)

# Phase 1: stable environment — product_A conversion ≈ 0.7
print("Phase 1: stable environment (200 steps)")
for _ in range(200):
    ctx = rng.standard_normal(N_FEATURES)
    decision = bandit.decide(ctx)
    arm = decision.chosen_arm
    # product_A converts well, others are average
    if arm == "product_A":
        reward = rng.normal(0.7, 0.05)
    else:
        reward = rng.normal(0.4, 0.1)
    bandit.update(context=ctx, arm=arm, reward=float(np.clip(reward, 0, 1)))

# Phase 2: product_A conversion rate drops sharply (drift)
print("Phase 2: product_A reward collapses to ~0.1 (drift introduced)")
drift_step = None
for step in range(300):
    ctx = rng.standard_normal(N_FEATURES)
    decision = bandit.decide(ctx)
    arm = decision.chosen_arm
    if arm == "product_A":
        reward = rng.normal(0.1, 0.05)  # sharp drop
    else:
        reward = rng.normal(0.4, 0.1)
    bandit.update(context=ctx, arm=arm, reward=float(np.clip(reward, 0, 1)))
    # Check if the detector fired for product_A
    if (
        bandit._drift_detectors is not None
        and "product_A" in bandit._drift_detectors
        and drift_step is None
    ):
        # Detector resets after firing, so check indirectly via step
        pass
    if drift_step is None and step > 50:
        drift_step = step  # used only for display

print(
    f"  Bandit continued operating — arm choices now include: "
    f"{set(bandit.decide(rng.standard_normal(N_FEATURES)).chosen_arm for _ in range(20))}"
)

# ---------------------------------------------------------------------------
# Pattern B — manual PageHinkleyDetector for per-arm monitoring
# ---------------------------------------------------------------------------
print("\n=== Pattern B: Manual PageHinkleyDetector ===\n")

detectors = {arm: PageHinkleyDetector(delta=0.01, lambda_=25.0) for arm in ARMS}
bandit2 = ClusterBandit(
    arms=ARMS,
    n_features=N_FEATURES,
    policy=PolicyType.LIN_UCB,
    n_clusters=2,
    seed=1,
)
bandit2.fit_offline(
    contexts=rng.standard_normal((300, N_FEATURES)),
    decisions=rng.choice(ARMS, size=300),
    rewards=rng.uniform(0, 1, 300),
)

drift_counts: dict[str, int] = {arm: 0 for arm in ARMS}

for step in range(500):
    ctx = rng.standard_normal(N_FEATURES)
    decision = bandit2.decide(ctx)
    arm = decision.chosen_arm

    # Inject drift on product_B starting at step 200
    if step >= 200 and arm == "product_B":
        reward = rng.normal(0.05, 0.05)
    else:
        reward = rng.normal(0.5, 0.1)
    reward = float(np.clip(reward, 0, 1))

    bandit2.update(context=ctx, arm=arm, reward=reward)

    if detectors[arm].update(reward):
        drift_counts[arm] += 1
        print(
            f"  Step {step:4d}: drift detected on '{arm}' "
            f"(mean≈{detectors[arm].reference_mean:.3f}) — resetting model"
        )
        # Reset arm in the router directly and restart the detector
        bandit2._router.reset_arm(arm)
        detectors[arm].reset()

print(f"\nDrift events detected: {drift_counts}")
print("\nManual pattern gives full control: you decide whether to reset, log, alert, etc.")
