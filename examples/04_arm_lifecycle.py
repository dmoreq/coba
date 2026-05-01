"""
04_arm_lifecycle.py — Dynamic arm lifecycle management.

Add new arms mid-flight (warm-started from existing arms), remove
underperformers, and control per-arm adaptation rates with gamma.

🚀 Run: streamlit run examples/04_arm_lifecycle.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from coba.types import PolicyType
from examples._shared import BanditSimulator, generate_linear_synthetic

st.set_page_config(page_title="coba Arm Lifecycle", layout="wide")

# ---- Initialise session state ----
if "sim" not in st.session_state:
    st.session_state.sim = None  # BanditSimulator
if "running" not in st.session_state:
    st.session_state.running = False
if "step_count" not in st.session_state:
    st.session_state.step_count = 0
if "max_steps" not in st.session_state:
    st.session_state.max_steps = 400
if "history_mean" not in st.session_state:
    st.session_state.history_mean: list[float] = []
if "history_events" not in st.session_state:
    st.session_state.history_events: list[str] = []
if "arm_snapshots" not in st.session_state:
    st.session_state.arm_snapshots: list[list[str]] = []

# ---- Sidebar ----
with st.sidebar:
    st.title("⚙️ Setup")
    st.markdown("---")

    n_features = st.slider("Features (d)", 3, 12, 4)
    n_clusters = st.slider("Clusters", 1, 6, 2)
    max_steps = st.slider("Total Steps", 100, 1000, 400, 50)

    st.markdown("---")
    st.markdown("### 🦾 Arm Actions")

    new_arm_name = st.text_input("New Arm Name", value="premium")
    warm_from = st.selectbox(
        "Warm-start from",
        ["(best current arm)"] + (st.session_state.sim.bandit.arms if st.session_state.sim else []),
    )
    new_arm_gamma = st.slider(
        "Gamma for new arm",
        0.5,
        1.0,
        0.95,
        0.01,
        help="Lower = faster adaptation for new arm",
    )

    c1, c2 = st.columns(2)
    add_clicked = c1.button("➕ Add Arm", use_container_width=True)
    remove_arm_name = c2.selectbox(
        "Remove Arm",
        st.session_state.sim.bandit.arms if st.session_state.sim else [],
    )
    remove_clicked = c2.button("➖ Remove Arm", use_container_width=True)

    st.markdown("---")
    if st.button("⏯️ Initialize / Reset", type="primary", use_container_width=True):
        st.session_state.running = True
        st.session_state.step_count = 0
        st.session_state.max_steps = max_steps
        st.session_state.history_mean = []
        st.session_state.history_events = []
        st.session_state.arm_snapshots = []
        st.session_state.sim = None
        st.rerun()

# ---- Main ----
st.title("🦾 Arm Lifecycle Manager")
st.markdown(
    """
Manage arms dynamically while the bandit is running. Add new arms warm-started
from existing models, set per-arm adaptation rates (gamma), and retire
underperformers.
"""
)

chart_placeholder = st.empty()
table_placeholder = st.empty()

if st.session_state.running:
    # Initialise simulator on first run
    if st.session_state.sim is None:
        arms = ["basic", "standard", "deluxe"]
        contexts, _, true_fn = generate_linear_synthetic(
            n_contexts=max_steps * 2, n_features=n_features, n_arms=3
        )
        sim = BanditSimulator(
            arms=arms,
            n_features=n_features,
            policy=PolicyType.LIN_UCB,
            n_clusters=n_clusters,
            alpha=0.5,
        )
        sim.bootstrap(contexts[: min(100, len(contexts) // 3)])
        st.session_state.sim = sim
        st.session_state.contexts = contexts
        st.session_state.true_fn = true_fn

    sim = st.session_state.sim

    # Handle add arm
    if add_clicked and new_arm_name not in sim.bandit.arms:
        warm_from_arm = warm_from if warm_from != "(best current arm)" else None
        sim.bandit.add_arm(new_arm_name, warm_start_from=warm_from_arm, gamma=new_arm_gamma)
        st.session_state.history_events.append(f"➕ Added '{new_arm_name}' (γ={new_arm_gamma})")
        st.rerun()

    # Handle remove arm
    if remove_clicked and remove_arm_name in sim.bandit.arms:
        sim.bandit.remove_arm(remove_arm_name)
        st.session_state.history_events.append(f"➖ Removed '{remove_arm_name}'")
        st.rerun()

    # Run one step
    step = st.session_state.step_count
    if step < st.session_state.max_steps:
        ctx = st.session_state.contexts[step]
        true_fn = st.session_state.true_fn
        decision = sim.bandit.decide(ctx)
        reward = true_fn(decision.chosen_arm, ctx)
        sim.bandit.update(context=ctx, arm=decision.chosen_arm, reward=reward)

        st.session_state.step_count = step + 1
        cumulative = sim.bandit.get_stats()
        if cumulative:
            mean_rew = np.mean([s.mean_reward for s in cumulative if s.n_pulls > 0])
        else:
            mean_rew = 0.0
        st.session_state.history_mean.append(mean_rew)

        # Snapshot arms every 20 steps
        if step % 20 == 0:
            st.session_state.arm_snapshots.append(list(sim.bandit.arms))

    # --- Chart ---
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(st.session_state.history_mean) + 1)),
            y=st.session_state.history_mean,
            mode="lines",
            name="Mean Arm Reward",
            line=dict(color="#636EFA", width=2),
        )
    )
    # Mark events
    for i, event in enumerate(st.session_state.history_events):
        x_pos = min((i + 1) * 30, len(st.session_state.history_mean))
        if x_pos > 0:
            fig.add_vline(
                x=x_pos,
                line_dash="dot",
                line_color=("#00CC96" if "Added" in event else "#EF553B"),
                annotation_text=event.split("'")[0] if "'" in event else event,
            )
    fig.update_layout(
        title="Cumulative Mean Arm Reward",
        xaxis_title="Step",
        yaxis_title="Mean Reward",
        yaxis_range=[0, 1],
        template="plotly_white",
    )
    chart_placeholder.plotly_chart(fig, use_container_width=True)

    # --- Arm stats table ---
    stats = sim.bandit.get_stats()
    rows = [
        {
            "Arm": s.arm,
            "Pulls": s.n_pulls,
            "Mean Reward": f"{s.mean_reward:.4f}",
            "Gamma": "custom" if hasattr(s, "gamma") and s.gamma else "default",
        }
        for s in stats
    ]
    table_placeholder.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Progress
    progress_val = min(1.0, (step + 1) / st.session_state.max_steps)
    st.progress(progress_val)
    st.caption(
        f"Step {step + 1}/{st.session_state.max_steps} — " f"Current arms: {sim.bandit.arms}"
    )

    # Auto-advance
    if step < st.session_state.max_steps - 1:
        st.rerun()

    # Educational
    with st.expander("📖 Arm Lifecycle Explained", expanded=True):
        st.markdown(
            r"""
### Dynamic Arm Management

**Warm-start** copies the model weights from an existing arm to the new one.
This avoids the "cold start" problem where a new arm starts with zero
information and needs hundreds of pulls to become competitive.

**Per-arm gamma** controls the forgetting rate:
$$
A_t = \gamma A_{t-1} + x_t x_t^\top, \quad
b_t = \gamma b_{t-1} + r_t x_t
$$
A lower gamma (e.g., 0.95) means the arm adapts faster to recent data —
useful for newly introduced arms or non-stationary environments.

**Removing arms** is safe anytime. The bandit simply stops considering them.
"""
        )
else:
    st.info("👈 Click **Initialize / Reset** to start the simulation.")
