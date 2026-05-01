"""
02_policy_lab.py — Side-by-side policy comparison.

Select 2-4 policies and watch them race on the same data. Which policy
learns fastest? Which converges to the highest reward?

🚀 Run: streamlit run examples/02_policy_lab.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from coba.types import PolicyType
from examples._shared import (
    BanditSimulator,
    generate_linear_synthetic,
    generate_nonlinear_synthetic,
)

st.set_page_config(page_title="coba Policy Lab", layout="wide")

# ---- Sidebar ----
with st.sidebar:
    st.title("⚙️ Setup")
    st.markdown("---")

    scenario = st.selectbox(
        "Reward Scenario",
        ["Linear (LinUCB should excel)", "Non-Linear (NeuralLinear should excel)"],
    )
    n_features = st.slider("Number of Features (d)", 3, 16, 6)
    n_arms = st.slider("Number of Arms (K)", 2, 8, 4)
    n_clusters = st.slider("Number of Clusters", 1, 6, 2)
    max_steps = st.slider("Simulation Steps", 100, 2000, 500, 100)
    animation_speed = st.select_slider("Speed", ["Fast", "Medium", "Slow"], value="Fast")
    speed_map = {"Fast": 0.0, "Medium": 0.02, "Slow": 0.08}
    _ = speed_map[animation_speed]  # Reserved for future animated rendering

    st.markdown("---")
    st.markdown("### 🏇 Select Policies to Race")
    selected_policies = st.multiselect(
        "Choose 2-4 policies",
        [p.value for p in PolicyType],
        default=[
            PolicyType.LIN_UCB.value,
            PolicyType.LIN_TS.value,
            PolicyType.THOMPSON.value,
        ],
        max_selections=4,
    )
    run = st.button("🏁 Start Race!", type="primary", use_container_width=True)

# ---- Main ----
st.title("🏇 Policy Comparison Lab")
st.markdown(
    """
Run **multiple bandit policies** on the same data and watch their cumulative
reward curves race in real-time. This helps you choose the right policy for
your use case.
"""
)

chart_placeholder = st.empty()
stats_placeholder = st.empty()

if run and len(selected_policies) < 2:
    st.warning("Please select at least 2 policies to compare.")
elif run:
    # Generate data
    if "Non-Linear" in scenario:
        contexts, arms, true_fn = generate_nonlinear_synthetic(
            n_contexts=max_steps, n_features=n_features, n_arms=n_arms
        )
    else:
        contexts, arms, true_fn = generate_linear_synthetic(
            n_contexts=max_steps, n_features=n_features, n_arms=n_arms
        )

    # Create simulators for each policy
    simulators: dict[str, BanditSimulator] = {}
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
    for i, p_name in enumerate(selected_policies):
        policy = PolicyType(p_name)
        sim = BanditSimulator(
            arms=arms,
            n_features=n_features,
            policy=policy,
            n_clusters=n_clusters,
            alpha=0.5,
            seed=42 + i,
        )
        sim.bootstrap(contexts[:80])
        simulators[p_name] = sim

    # Initialize chart
    fig = go.Figure()
    for p_name, color in zip(selected_policies, colors):
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                name=p_name,
                line=dict(color=color, width=2),
            )
        )
    fig.add_hline(
        y=1.0 / n_arms,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Random (1/{n_arms})",
    )
    fig.update_layout(
        title="Cumulative Mean Reward — Policy Race",
        xaxis_title="Step",
        yaxis_title="Cumulative Mean Reward",
        yaxis_range=[0, 1],
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1),
    )

    # Run all simulators in lockstep
    for step_idx in range(max_steps):
        ctx = contexts[step_idx]

        for i, p_name in enumerate(selected_policies):
            rec = next(
                simulators[p_name].run(
                    max_steps=1,
                    contexts=np.array([ctx]),
                    true_reward_fn=true_fn,
                    sleep_s=0,
                )
            )
            fig.data[i].x = list(fig.data[i].x) + [step_idx + 1]
            fig.data[i].y = list(fig.data[i].y) + [rec.cumulative_mean_reward]

        # Render every 5 steps for performance
        if step_idx % 5 == 0 or step_idx == max_steps - 1:
            chart_placeholder.plotly_chart(fig, use_container_width=True)

            # Stats table
            rows = []
            for p_name in selected_policies:
                sim = simulators[p_name]
                rows.append(
                    {
                        "Policy": p_name,
                        "Final Mean Reward": f"{sim.cumulative_mean_reward:.4f}",
                        "Cumulative Regret": f"{sim.cumulative_regret:.2f}",
                    }
                )
            stats_placeholder.dataframe(
                pd.DataFrame(rows).set_index("Policy"),
                use_container_width=True,
            )

    # Final summary
    best_policy = max(simulators, key=lambda p: simulators[p].cumulative_mean_reward)
    st.success(
        f"🏆 **{best_policy}** wins with mean reward "
        f"**{simulators[best_policy].cumulative_mean_reward:.4f}**"
    )

    # Educational section
    with st.expander("📖 Understanding the Policies", expanded=True):
        st.markdown(
            r"""
### Policy Types Explained

| Policy | Mechanism | Best For |
|--------|-----------|----------|
| **LinUCB** | Ridge regression + UCB bonus | Linear rewards, moderate dimensions |
| **LinTS** | Ridge regression + Thompson Sampling | Linear rewards, better exploration |
| **Thompson** | Context-free Beta-Bernoulli | Simple scenarios, no features |
| **UCB1** | Context-free Hoeffding bound | Baseline, arms without context |
| **Logistic UCB/TS** | Logistic link for binary rewards | Click/convert prediction |
| **NeuralLinear** | MLP embedding + LinTS | Non-linear reward functions |
| **Epsilon-Greedy** | Explores randomly ε% of the time | Simple, interpretable |
| **Bootstrapped TS/UCB** | Ensemble of sklearn models | Complex non-linear patterns |

**Why do policies differ?** LinUCB uses an *optimistic* bonus proportional
to uncertainty. Thompson Sampling *samples* from the posterior and picks
the best. In theory, TS often explores more efficiently, but UCB can be
more stable in practice.
"""
        )
else:
    st.info("👈 Select at least 2 policies in the sidebar and click **Start Race!**")
