"""
05_production_ops.py — Production operations dashboard.

Covers persistence, top-K decisions, reward normalization, abstention
thresholds, and minimum pull-rate constraints.

🚀 Run: streamlit run examples/05_production_ops.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from coba import ClusterBandit, RewardNormalizer
from coba.persistence import load_bandit, save_bandit
from coba.types import PolicyType
from examples._shared import generate_linear_synthetic

st.set_page_config(page_title="coba Production Ops", layout="wide")

st.title("🏭 Production Operations")
st.markdown(
    "Utilities for running coba in production: persistence, top-K, "
    "normalization, abstention, and constraints."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "💾 Persistence",
        "☝️ Top-K",
        "📐 Normalizer",
        "🤚 Abstention",
        "🔒 Constraints",
    ]
)


# ---- Shared setup ----
@st.cache_resource
def get_bandit_and_data():
    """Return a trained bandit and contexts for production ops demos."""
    arms = ["tier_1", "tier_2", "tier_3", "tier_4", "tier_5"]
    ctx, _, _ = generate_linear_synthetic(n_contexts=500, n_features=4, n_arms=5)
    bandit = ClusterBandit(
        arms=arms,
        n_features=4,
        policy=PolicyType.LIN_UCB,
        n_clusters=3,
        seed=0,
    )
    boot_ctx = ctx[:200]
    rng = np.random.default_rng(0)
    boot_dec = rng.choice(arms, size=200)
    boot_rew = np.clip(rng.normal(0.5, 0.2, size=200), 0, 1)
    bandit.fit_offline(boot_ctx, boot_dec, boot_rew)
    return bandit, ctx


bandit, contexts = get_bandit_and_data()

# ============================================================================
# Tab 1: Persistence
# ============================================================================
with tab1:
    st.subheader("💾 Save & Reload")
    st.markdown("Save a trained bandit to disk and reload it — decisions " "must be identical.")

    test_ctx = contexts[300]
    original = bandit.decide(test_ctx)
    st.write(f"**Original**: arm=`{original.chosen_arm}`, " f"score=`{original.score:.4f}`")

    if st.button("💾 Save & Reload", key="persist_btn"):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bandit.joblib"
            save_bandit(bandit, path)
            loaded = load_bandit(path)
            loaded_decision = loaded.decide(test_ctx)
            st.write(
                f"**Reloaded**: arm=`{loaded_decision.chosen_arm}`, "
                f"score=`{loaded_decision.score:.4f}`"
            )
            assert original.chosen_arm == loaded_decision.chosen_arm
            st.success("✅ Round-trip verified — decisions match!")

    with st.expander("📖 Usage", expanded=False):
        st.code(
            """\
from coba.persistence import save_bandit, load_bandit

# Save
save_bandit(bandit, "model.joblib")

# Load
bandit = load_bandit("model.joblib")
decision = bandit.decide(context)
""",
            language="python",
        )

# ============================================================================
# Tab 2: Top-K
# ============================================================================
with tab2:
    st.subheader("☝️ Ranked Top-K Decisions")
    st.markdown("Get a ranked list of arms instead of just the single best.")

    test_ctx2 = contexts[301]
    k = st.slider("K", 1, len(bandit.arms), 3, key="topk_k")
    top_k = bandit.decide_top_k(test_ctx2, k=k)

    fig = go.Figure(
        go.Bar(
            x=[arm for arm, _ in top_k],
            y=[score for _, score in top_k],
            marker_color=[
                "#636EFA",
                "#00CC96",
                "#AB63FA",
                "#FFA15A",
                "#19D3F3",
            ][:k],
        )
    )
    fig.update_layout(
        title=f"Top-{k} Arms by Score",
        yaxis_title="Score",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.code(
        """\
top3 = bandit.decide_top_k(context, k=3)
for rank, (arm, score) in enumerate(top3, start=1):
    print(f"#{rank}: {arm} ({score:.4f})")
""",
        language="python",
    )

# ============================================================================
# Tab 3: Reward Normalizer
# ============================================================================
with tab3:
    st.subheader("📐 Reward Normalization")
    st.markdown(
        "Some policies (Thompson, Logistic) require rewards in $[0, 1]$. "
        "`RewardNormalizer` scales raw business metrics online."
    )

    mode = st.selectbox("Mode", ["minmax", "zscore"])
    decay = st.slider("Decay (EWMA)", 0.9, 1.0, 0.999, 0.001)
    normalizer = RewardNormalizer(mode=mode, decay=decay)

    raw_values = [100.0, 250.0, 50.0, 500.0, 300.0, 20.0, 800.0, 150.0]
    normed = []
    for raw in raw_values:
        normed.append(normalizer.update_and_normalize(raw))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(raw_values) + 1)),
            y=raw_values,
            mode="lines+markers",
            name="Raw",
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(normed) + 1)),
            y=normed,
            mode="lines+markers",
            name="Normalized",
            yaxis="y2",
        )
    )
    fig.update_layout(
        yaxis=dict(title="Raw Value"),
        yaxis2=dict(title="Normalized", overlaying="y", side="right"),
        template="plotly_white",
        title="Online Normalization",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.code(
        """\
normalizer = RewardNormalizer(mode="minmax", decay=0.999)
normed = normalizer.update_and_normalize(raw_reward)
bandit.update(context, arm, reward=normed)
""",
        language="python",
    )

# ============================================================================
# Tab 4: Abstention
# ============================================================================
with tab4:
    st.subheader("🤚 Confidence-Based Abstention")
    st.markdown(
        "When the top two arms are too close in score, the bandit can "
        "**abstain** and fall back to a safe default."
    )

    gap = st.slider("Min Confidence Gap", 0.0, 0.5, 0.05, 0.01)
    n_test = st.slider("Test contexts", 50, 500, 200, 50, key="abs_n")

    decided = 0
    abstained = 0
    for i in range(n_test):
        d = bandit.decide(contexts[i], min_confidence_gap=gap)
        if d.abstained:
            abstained += 1
        else:
            decided += 1

    fig = go.Figure(
        go.Pie(
            labels=["Decided", "Abstained"],
            values=[decided, abstained],
            hole=0.5,
            marker_colors=["#636EFA", "#EF553B"],
        )
    )
    fig.update_layout(
        title=f"Abstention Rate (gap ≥ {gap})",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.metric(
        "Abstention Rate",
        f"{abstained / n_test:.1%}" if n_test else "N/A",
    )

    st.code(
        """\
decision = bandit.decide(context, min_confidence_gap=0.05)
if decision.abstained:
    arm = fallback_arm
else:
    arm = decision.chosen_arm
""",
        language="python",
    )

# ============================================================================
# Tab 5: Constraints
# ============================================================================
with tab5:
    st.subheader("🔒 Minimum Pull-Rate Constraints")
    st.markdown("Force guaranteed exploration — e.g., contractual or A/B test " "requirements.")

    c_arms = ["hero", "sponsored", "editorial", "new_feature"]
    c_bandit = ClusterBandit(
        arms=c_arms,
        n_features=4,
        policy=PolicyType.LIN_UCB,
        n_clusters=2,
        seed=0,
        min_pull_rates={"sponsored": 0.10, "new_feature": 0.05},
    )
    rng_c = np.random.default_rng(0)
    c_bandit.fit_offline(
        contexts[:200],
        rng_c.choice(c_arms, size=200),
        np.clip(rng_c.normal(0.5, 0.2, size=200), 0, 1),
    )

    n_steps_c = st.slider("Steps", 100, 1000, 500, 50, key="constraint_n")
    if st.button("Run Constrained Simulation", key="constraint_btn"):
        counts = {a: 0 for a in c_arms}
        for i in range(n_steps_c):
            d = c_bandit.decide(contexts[i])
            counts[d.chosen_arm] += 1
            c_bandit.update(contexts[i], d.chosen_arm, 0.5)

        total = sum(counts.values()) or 1
        rows = []
        for arm in c_arms:
            rate = counts[arm] / total
            required: float | None = {
                "sponsored": 0.10,
                "new_feature": 0.05,
            }.get(arm)
            status = "✅" if required is None or rate >= required - 0.01 else "❌"
            rows.append(
                {
                    "Arm": arm,
                    "Pulls": counts[arm],
                    "Rate": f"{rate:.1%}",
                    "Constraint": f"≥ {required:.0%}" if required else "—",
                    "Status": status,
                }
            )

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        all_ok = all(r["Status"] == "✅" for r in rows)
        if all_ok:
            st.success("All constraints satisfied!")
        else:
            st.warning("Some constraints violated — increase simulation steps.")

    st.code(
        """\
bandit = ClusterBandit(
    arms=["hero", "sponsored", "new_feature"],
    min_pull_rates={"sponsored": 0.10, "new_feature": 0.05},
    ...
)
""",
        language="python",
    )
