"""
01_playground.py — Interactive contextual bandit playground.

Adjust parameters in the sidebar, click "Run Simulation", and watch
the bandit learn in real-time with animated charts and explanations.

🚀 Run: streamlit run examples/01_playground.py
"""

from __future__ import annotations

import streamlit as st

from coba.types import PolicyType
from examples._shared import (
    BanditSimulator,
    create_arm_pull_chart,
    create_arm_score_chart,
    create_cluster_distribution_chart,
    create_regret_chart,
    create_reward_chart,
    generate_linear_synthetic,
    update_arm_pull_chart,
    update_arm_score_chart,
    update_cluster_chart,
    update_regret_chart,
    update_reward_chart,
)

st.set_page_config(page_title="coba Playground", layout="wide")

# ---- Sidebar Controls ----
with st.sidebar:
    st.title("🎮 Controls")
    st.markdown("---")

    n_features = st.slider("Number of Features (d)", 3, 16, 5)
    n_arms = st.slider("Number of Arms (K)", 2, 10, 4)
    n_clusters = st.slider("Number of Clusters", 1, 8, 3)

    policy_name = st.selectbox(
        "Policy",
        [p.value for p in PolicyType],
        index=0,
    )
    policy = PolicyType(policy_name)

    alpha = st.slider("Alpha (exploration bonus)", 0.1, 3.0, 0.5, 0.1)

    max_steps = st.slider("Simulation Steps", 50, 1000, 300, 50)
    animation_speed = st.select_slider(
        "Animation Speed",
        options=["Fast ⚡", "Medium 🏃", "Slow 🐢"],
        value="Fast ⚡",
    )
    speed_map = {"Fast ⚡": 0.0, "Medium 🏃": 0.03, "Slow 🐢": 0.1}
    sleep_s = speed_map[animation_speed]

    run = st.button("▶️ Run Simulation", type="primary", use_container_width=True)

# ---- Main Display ----
st.title("🕹️ Contextual Bandit Playground")
st.markdown(
    """
Watch a **cluster-based contextual bandit** learn in real-time.
The bandit decides which arm to pull for each incoming context,
observes a reward, and updates its models — all while charts animate live.
"""
)

# --- Layout: two columns of charts ---
col1, col2 = st.columns(2)
reward_placeholder = col1.empty()
regret_placeholder = col2.empty()

col3, col4, col5 = st.columns(3)
score_placeholder = col3.empty()
cluster_placeholder = col4.empty()
pull_placeholder = col5.empty()

# Status bar
status_placeholder = st.empty()

if run:
    # Generate synthetic data
    with st.spinner("Generating synthetic data..."):
        contexts, arms, true_fn = generate_linear_synthetic(
            n_contexts=max_steps * 2, n_features=n_features, n_arms=n_arms
        )

    # Initialize simulator
    sim = BanditSimulator(
        arms=arms,
        n_features=n_features,
        policy=policy,
        n_clusters=n_clusters,
        alpha=alpha,
    )

    # Bootstrap
    with st.spinner("Bootstrapping bandit from historical data..."):
        sim.bootstrap(contexts[: min(100, len(contexts) // 3)])

    # Initialize charts
    reward_fig = create_reward_chart(n_arms=n_arms)
    regret_fig = create_regret_chart()
    score_fig = create_arm_score_chart(arms)
    cluster_fig = create_cluster_distribution_chart(n_clusters)
    pull_fig = create_arm_pull_chart(arms)

    # Initialize rolling buffers (keep last 200 points for readability)
    steps_buffer: list[int] = []
    rewards_buffer: list[float] = []
    regrets_buffer: list[float] = []
    cluster_counts: dict[int, int] = {i: 0 for i in range(n_clusters)}
    pull_counts: dict[str, int] = {a: 0 for a in arms}

    # Run simulation with live updates
    for step_rec in sim.run(
        max_steps=max_steps,
        contexts=contexts,
        true_reward_fn=true_fn,
        sleep_s=sleep_s,
    ):
        # Update buffers (keep last 200)
        steps_buffer.append(step_rec.step)
        rewards_buffer.append(step_rec.cumulative_mean_reward)
        regrets_buffer.append(step_rec.cumulative_regret)
        if len(steps_buffer) > 200:
            steps_buffer = steps_buffer[-200:]
            rewards_buffer = rewards_buffer[-200:]
            regrets_buffer = regrets_buffer[-200:]

        # Update cluster counts
        if step_rec.cluster >= 0:
            cluster_counts[step_rec.cluster] = cluster_counts.get(step_rec.cluster, 0) + 1

        # Update pull counts
        pull_counts[step_rec.chosen_arm] += 1

        # Update charts
        reward_fig = update_reward_chart(reward_fig, steps_buffer, rewards_buffer)
        regret_fig = update_regret_chart(regret_fig, steps_buffer, regrets_buffer)
        score_fig = update_arm_score_chart(score_fig, step_rec.all_scores)
        cluster_fig = update_cluster_chart(cluster_fig, cluster_counts)
        pull_fig = update_arm_pull_chart(pull_fig, pull_counts)

        # Render
        reward_placeholder.plotly_chart(
            reward_fig, use_container_width=True, key=f"rw_{step_rec.step}"
        )
        regret_placeholder.plotly_chart(
            regret_fig, use_container_width=True, key=f"rg_{step_rec.step}"
        )
        score_placeholder.plotly_chart(
            score_fig, use_container_width=True, key=f"sc_{step_rec.step}"
        )
        cluster_placeholder.plotly_chart(
            cluster_fig, use_container_width=True, key=f"cl_{step_rec.step}"
        )
        pull_placeholder.plotly_chart(pull_fig, use_container_width=True, key=f"pl_{step_rec.step}")

        # Status
        status_placeholder.markdown(
            f"**Step {step_rec.step}/{max_steps}** | "
            f"Chose: `{step_rec.chosen_arm}` | "
            f"Reward: {step_rec.reward:.3f} | "
            f"Mean: {step_rec.cumulative_mean_reward:.3f} | "
            f"Regret: {step_rec.cumulative_regret:.2f}"
        )

    # Final summary
    st.success(
        f"✅ Simulation complete! Final cumulative mean reward: "
        f"**{sim.cumulative_mean_reward:.4f}** "
        f"(random baseline: {1.0 / n_arms:.4f})"
    )

    # ---- Educational Section ----
    with st.expander("📖 How It Works", expanded=True):
        st.markdown(
            r"""
### What just happened?

The **ClusterBandit** works in three phases:

1. **Clustering** — Each incoming context is assigned to one of $K$ clusters
   via KMeans. This partitions the context space so similar contexts get similar
   treatment.

2. **Arm Scoring** — Within the assigned cluster, each arm maintains its own
   model (in this case, a **LinUCB** ridge regression). The model predicts the
   expected reward plus an *exploration bonus* proportional to uncertainty:
   $$
   \text{score}_a = \hat{\theta}_a^\top x + \alpha \sqrt{x^\top A_a^{-1} x}
   $$

3. **Decision** — The arm with the highest score is chosen (greedy upper
   confidence bound). The choice balances **exploitation** (high expected
   reward) vs **exploration** (high uncertainty).

**Regret** measures how much reward we lost compared to always picking the
optimal arm. A good bandit has regret that grows sub-linearly ($O(\sqrt{T})$).
"""
        )

else:
    # Show placeholder before first run
    st.info("👈 Adjust parameters in the sidebar and click **Run Simulation** to begin!")
