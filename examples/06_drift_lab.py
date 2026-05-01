"""
06_drift_lab.py — Concept drift detection and recovery.

Inject concept drift mid-simulation and watch the Page-Hinkley detector
fire. The affected arm is automatically reset and re-learns from scratch.

🚀 Run: streamlit run examples/06_drift_lab.py
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from coba import ClusterBandit, PageHinkleyDetector
from coba.types import PolicyType

st.set_page_config(page_title="coba Drift Lab", layout="wide")

# ---- Sidebar ----
with st.sidebar:
    st.title("⚙️ Setup")
    st.markdown("---")

    n_features = st.slider("Features (d)", 3, 12, 4)
    n_clusters = st.slider("Clusters", 1, 6, 2)
    total_steps = st.slider("Total Steps", 200, 1000, 500, 50)

    delta = st.slider(
        "PH Delta (detection sensitivity)",
        0.001,
        0.05,
        0.005,
        0.001,
        help="Minimum detectable change in reward mean",
    )
    lambda_ = st.slider(
        "PH Lambda (threshold)",
        10.0,
        100.0,
        30.0,
        5.0,
        help="Lower = more sensitive, faster detection",
    )

    st.markdown("---")
    st.markdown("### 💥 Drift Injection")
    drift_arm = st.selectbox("Affected Arm", ["product_A", "product_B", "product_C"])
    drift_from = st.slider("Drift reward drops from", 0.3, 0.9, 0.7, 0.05)
    drift_to = st.slider("to", 0.0, 0.3, 0.1, 0.05)
    drift_step = st.slider("Drift starts at step", 50, total_steps - 50, 150, 10)

    auto_mode = st.checkbox(
        "Automatic drift detection",
        value=True,
        help="ClusterBandit handles reset automatically",
    )

    run = st.button("▶️ Run Simulation", type="primary", use_container_width=True)

# ---- Main ----
st.title("💥 Concept Drift Laboratory")
st.markdown(
    r"""
Simulate a real-world scenario where an arm's reward distribution suddenly
changes (e.g., product goes out of stock, user preferences shift).

The **Page-Hinkley** detector monitors the reward stream and triggers
a model reset when drift is detected.
"""
)

if run:
    ARMS = ["product_A", "product_B", "product_C"]
    rng = np.random.default_rng(42)

    # Create bandit
    if auto_mode:
        bandit = ClusterBandit(
            arms=ARMS,
            n_features=n_features,
            policy=PolicyType.LIN_UCB,
            n_clusters=n_clusters,
            seed=0,
            enable_drift_detection=True,
            drift_delta=delta,
            drift_lambda=lambda_,
        )
    else:
        bandit = ClusterBandit(
            arms=ARMS,
            n_features=n_features,
            policy=PolicyType.LIN_UCB,
            n_clusters=n_clusters,
            seed=0,
        )
        detectors = {arm: PageHinkleyDetector(delta=delta, lambda_=lambda_) for arm in ARMS}

    # Bootstrap
    bandit.fit_offline(
        contexts=rng.standard_normal((200, n_features)),
        decisions=rng.choice(ARMS, size=200),
        rewards=rng.uniform(0, 1, 200),
    )

    # Run simulation
    rewards_history: list[float] = []
    arms_chosen: list[str] = []
    drift_events: list[int] = []

    chart_placeholder = st.empty()
    status_placeholder = st.empty()
    dist_placeholder = st.empty()

    for step in range(total_steps):
        ctx = rng.standard_normal(n_features)
        decision = bandit.decide(ctx)
        arm = decision.chosen_arm

        # Inject drift
        if step >= drift_step and arm == drift_arm:
            reward = float(np.clip(rng.normal(drift_to, 0.05), 0, 1))
        elif arm == drift_arm:
            reward = float(np.clip(rng.normal(drift_from, 0.05), 0, 1))
        else:
            reward = float(np.clip(rng.normal(0.4, 0.1), 0, 1))

        bandit.update(context=ctx, arm=arm, reward=reward)

        # Manual detection
        if not auto_mode and arm in detectors:
            if detectors[arm].update(reward):
                drift_events.append(step)
                bandit._router.reset_arm(arm)
                detectors[arm].reset()

        rewards_history.append(reward)
        arms_chosen.append(str(arm))

        # Render every 10 steps
        if step % 10 == 0 or step == total_steps - 1:
            # Cumulative mean reward
            cum_mean = np.cumsum(rewards_history) / (np.arange(len(rewards_history)) + 1)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(cum_mean) + 1)),
                    y=cum_mean,
                    mode="lines",
                    name="Mean Reward",
                    line=dict(color="#636EFA", width=2),
                )
            )
            # Mark drift start
            fig.add_vline(
                x=drift_step + 1,
                line_dash="dash",
                line_color="red",
                annotation_text="Drift injected",
            )
            # Mark detection events
            for ev in drift_events:
                fig.add_vline(
                    x=ev + 1,
                    line_dash="dot",
                    line_color="green",
                    annotation_text="Reset",
                )
            fig.update_layout(
                title="Cumulative Mean Reward with Drift",
                yaxis_range=[0, 1],
                template="plotly_white",
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)

            status_placeholder.markdown(
                f"**Step {step + 1}/{total_steps}** | "
                f"Drift events detected: {len(drift_events)}"
            )

    # After simulation, show before/after reward distribution for drifted arm
    before_drift = [
        r for i, r in enumerate(rewards_history) if i < drift_step and arms_chosen[i] == drift_arm
    ]
    after_drift = [
        r for i, r in enumerate(rewards_history) if i >= drift_step and arms_chosen[i] == drift_arm
    ]

    fig_dist = go.Figure()
    if before_drift:
        fig_dist.add_trace(
            go.Histogram(
                x=before_drift,
                name="Before Drift",
                marker_color="#636EFA",
                opacity=0.7,
            )
        )
    if after_drift:
        fig_dist.add_trace(
            go.Histogram(
                x=after_drift,
                name="After Drift",
                marker_color="#EF553B",
                opacity=0.7,
            )
        )
    fig_dist.update_layout(
        title=f"Reward Distribution for '{drift_arm}'",
        barmode="overlay",
        template="plotly_white",
    )
    dist_placeholder.plotly_chart(fig_dist, use_container_width=True)

    # Educational
    with st.expander("📖 Page-Hinkley Detector Explained", expanded=True):
        st.markdown(
            r"""
### Page-Hinkley Change-Point Detection

The PH test monitors a running mean and triggers when the cumulative
deviation exceeds a threshold:

$$
m_T = \frac{1}{T}\sum_{t=1}^T r_t, \quad
S_T = \sum_{t=1}^T (r_t - m_t - \delta)
$$

When $S_T - \min_{t \le T} S_t > \lambda$, drift is detected at time $T$.

**Parameters:**
- $\delta$ (delta): Minimum change magnitude to detect. Larger = only big shifts.
- $\lambda$ (lambda): Detection threshold. Lower = faster but more false positives.

**When drift fires**, coba resets the affected arm's model (clears $A^{-1}$
matrix) so it can re-learn from scratch without being anchored to old data.
"""
        )

else:
    st.info("👈 Configure drift parameters and click **Run Simulation**.")
