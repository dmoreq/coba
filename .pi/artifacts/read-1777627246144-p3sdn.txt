"""
03_offline_eval.py — Offline policy evaluation and debiasing.

Learn how importance sampling corrects for biased historical data.
Compare IPS vs Doubly-Robust vs NCIS evaluation methods.

🚀 Run: streamlit run examples/03_offline_eval.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import Ridge

from coba import ClusterBandit
from coba.types import PolicyType
from examples._shared import generate_biased_log_data

st.set_page_config(page_title="coba Offline Eval", layout="wide")

# ---- Sidebar ----
with st.sidebar:
    st.title("⚙️ Setup")
    st.markdown("---")

    n_logs = st.slider("Logged Interactions", 200, 2000, 800, 100)
    n_features = st.slider("Number of Features (d)", 3, 12, 5)
    bias_level = st.selectbox(
        "Logging Policy Bias",
        ["Mild (60% on worst arm)", "Severe (90% on worst arm)"],
    )
    max_steps = st.slider("Online Test Steps", 50, 500, 200, 50)

    st.markdown("---")
    run = st.button("▶️ Run Evaluation", type="primary", use_container_width=True)

# ---- Main ----
st.title("📊 Offline Policy Evaluation")
st.markdown(
    r"""
When you have **historical log data** from an old policy (e.g., a rule-based
system), you need to correct for its biases before training a new bandit.

This demo shows how **importance sampling** corrects for a logging policy
that favoured the wrong arms.
"""
)

if run:
    arms = ["arm_A", "arm_B", "arm_C"]

    # Adjust bias severity
    bias_pct = 0.90 if "Severe" in bias_level else 0.70

    # Generate biased logs (logging policy favours arm_C, but arm_A is truly best)
    contexts, decisions, propensities, rewards = generate_biased_log_data(
        n_logs=n_logs, n_features=n_features, arms=arms
    )
    # Override bias level: the generator uses 70%, we scale if needed
    if bias_pct > 0.70:
        n_arms = len(arms)
        high_prob = bias_pct
        low_prob = (1.0 - high_prob) / (n_arms - 1)
        probs = [low_prob] * (n_arms - 1) + [high_prob]
        rng2 = np.random.default_rng(42)
        decisions = rng2.choice(arms, size=n_logs, p=probs)
        propensities = np.array([probs[arms.index(a)] for a in decisions])

    # Show logging policy distribution
    col_a, col_b = st.columns(2)
    with col_a:
        log_counts = pd.Series(decisions).value_counts()
        fig_log = go.Figure(go.Bar(x=list(log_counts.index), y=list(log_counts.values)))
        fig_log.update_layout(title="Logging Policy Decisions", template="plotly_white")
        st.plotly_chart(fig_log, use_container_width=True)
    with col_b:
        mean_by_arm = {}
        for a in arms:
            mask = decisions == a
            mean_by_arm[a] = float(rewards[mask].mean()) if mask.any() else 0.0
        fig_reward = go.Figure(go.Bar(x=list(mean_by_arm.keys()), y=list(mean_by_arm.values())))
        fig_reward.update_layout(
            title="True Mean Reward by Arm", yaxis_range=[0, 1], template="plotly_white"
        )
        st.plotly_chart(fig_reward, use_container_width=True)

    st.markdown("---")

    # --- IPS Bootstrap ---
    st.subheader("1. Pure IPS (Inverse Propensity Scoring)")
    bandit_ips = ClusterBandit(arms=arms, n_features=n_features, policy=PolicyType.LIN_UCB, seed=0)
    bandit_ips.fit_offline(contexts, decisions, rewards, propensities=propensities)

    # --- DR Bootstrap ---
    st.subheader("2. Doubly-Robust (IPS + reward model)")
    reward_model = Ridge(alpha=1.0).fit(contexts, rewards)
    reward_estimates = reward_model.predict(contexts)

    bandit_dr = ClusterBandit(arms=arms, n_features=n_features, policy=PolicyType.LIN_UCB, seed=0)
    bandit_dr.fit_offline(
        contexts,
        decisions,
        rewards,
        propensities=propensities,
        use_dr=True,
        reward_estimates=reward_estimates,
    )

    # --- Naive (no correction) ---
    st.subheader("3. Naive (no bias correction)")
    bandit_naive = ClusterBandit(
        arms=arms, n_features=n_features, policy=PolicyType.LIN_UCB, seed=0
    )
    bandit_naive.fit_offline(contexts, decisions, rewards)

    # --- Evaluate all three online ---
    st.markdown("---")
    st.subheader("📈 Online Performance Comparison")

    rng = np.random.default_rng(42)
    test_ctx = rng.standard_normal((max_steps, n_features))
    true_means: dict[str, float] = {"arm_A": 0.75, "arm_B": 0.3, "arm_C": 0.3}

    results: dict[str, list[float]] = {"IPS": [], "DR": [], "Naive": []}
    cum_ips = cum_dr = cum_naive = 0.0

    for t, ctx in enumerate(test_ctx):
        # Use LinUCB uncertainty, not true_means, for the actual reward
        d_ips = bandit_ips.decide(ctx)
        d_dr = bandit_dr.decide(ctx)
        d_naive = bandit_naive.decide(ctx)

        r_ips = float(np.clip(rng.normal(true_means[d_ips.chosen_arm], 0.1), 0, 1))
        r_dr = float(np.clip(rng.normal(true_means[d_dr.chosen_arm], 0.1), 0, 1))
        r_naive = float(np.clip(rng.normal(true_means[d_naive.chosen_arm], 0.1), 0, 1))

        cum_ips += r_ips
        cum_dr += r_dr
        cum_naive += r_naive
        results["IPS"].append(cum_ips / (t + 1))
        results["DR"].append(cum_dr / (t + 1))
        results["Naive"].append(cum_naive / (t + 1))

    fig = go.Figure()
    plot_colors = {"IPS": "#636EFA", "DR": "#00CC96", "Naive": "#EF553B"}
    for method, data in results.items():
        fig.add_trace(
            go.Scatter(
                x=list(range(1, max_steps + 1)),
                y=data,
                mode="lines",
                name=method,
                line=dict(color=plot_colors[method], width=2),
            )
        )
    fig.add_hline(
        y=0.75, line_dash="dash", line_color="gray", annotation_text="Optimal arm_A (0.75)"
    )
    fig.add_hline(y=0.3, line_dash="dot", line_color="gray", annotation_text="Suboptimal (0.30)")
    fig.update_layout(
        title="Cumulative Mean Reward — IPS vs DR vs Naive",
        xaxis_title="Step",
        yaxis_title="Mean Reward",
        yaxis_range=[0, 1],
        template="plotly_white",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Evaluation metrics
    st.markdown("---")
    st.subheader("📋 Evaluation Metrics")

    rs = bandit_ips.evaluate_rejection_sampling(contexts, decisions, rewards)
    st.markdown(
        f"**Rejection Sampling**: estimated reward = "
        f"{rs.estimated_reward:.4f}, CI = [{rs.ci_lower:.4f}, {rs.ci_upper:.4f}]"
    )

    dr_eval = bandit_ips.evaluate_doubly_robust(
        contexts,
        decisions,
        rewards,
        propensities,
        reward_estimates,
        target_reward_estimates=reward_estimates,
    )
    st.markdown(
        f"**Doubly-Robust**: estimated reward = "
        f"{dr_eval.estimated_reward:.4f}, CI = [{dr_eval.ci_lower:.4f}, {dr_eval.ci_upper:.4f}]"
    )

    policy_scores = np.array(
        [bandit_ips.score_all(c).get(a, 1e-6) for c, a in zip(contexts, decisions)]
    )
    ncis = bandit_ips.evaluate_ncis(
        policy_scores=policy_scores,
        logging_scores=propensities,
        rewards=rewards,
    )
    st.markdown(
        f"**NCIS**: estimated reward = "
        f"{ncis.estimated_reward:.4f}, CI = [{ncis.ci_lower:.4f}, {ncis.ci_upper:.4f}]"
    )

    # Educational section
    with st.expander("📖 How Off-Policy Correction Works", expanded=True):
        st.markdown(
            r"""
### Why Bias Correction Matters

When training from historical logs, the logging policy $\pi_0$ may favour
certain arms. If you naively treat the data as unbiased, the learned policy
$\pi$ inherits those biases.

**IPS (Inverse Propensity Scoring)** reweights each sample by:
$$
\mathbb{E}[R_\pi] \approx \frac{1}{n}\sum_{i=1}^n r_i \cdot
\frac{\mathbb{1}[a_i = \pi(x_i)]}{p_0(a_i \mid x_i)}
$$

The weight $1 / p_0(a_i \mid x_i)$ is large when the logging policy
was unlikely to pick that arm — it "corrects" for under-representation.

**Doubly-Robust** adds a reward model $\hat{r}(x, a)$ to reduce variance:
$$
\hat{V}_{DR} = \frac{1}{n}\sum_i
\left[\hat{r}(x_i,\pi(x_i)) +
\frac{\mathbb{1}[a_i = \pi(x_i)]}{p_0(a_i)} (r_i - \hat{r}(x_i,a_i))\right]
$$
"""
        )

else:
    st.info(
        "👈 Adjust parameters and click **Run Evaluation** to see off-policy methods in action."
    )
