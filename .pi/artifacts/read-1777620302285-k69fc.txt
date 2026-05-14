"""
Shared simulation engine for coba Streamlit examples.

Provides:
  - synthetic data generators (linear, non-linear, biased-log)
  - BanditSimulator: streaming decide -> update iterator with per-step metrics
  - Plotly chart factory functions for live-updating dashboard charts
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable, Generator

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from coba import ClusterBandit
from coba.types import PolicyType


# ---------------------------------------------------------------------------
# StepRecord & Simulation Metrics
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    """A single step of the bandit loop."""

    step: int
    context: np.ndarray
    chosen_arm: Any
    reward: float
    cumulative_mean_reward: float
    cumulative_regret: float
    all_scores: dict[Any, float]
    cluster: int


# ---------------------------------------------------------------------------
# Part A — Synthetic data generators
# ---------------------------------------------------------------------------


def generate_linear_synthetic(
    n_contexts: int = 500,
    n_features: int = 4,
    n_arms: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, list[str], Callable[[str, np.ndarray], float]]:
    """Generate linear-reward data with a known best arm.

    Returns ``(contexts, arms, true_reward_fn)``.

    ``true_reward_fn(arm, context) -> float in [0, 1]``.
    arm_0 is intentionally best (all-positive weights).
    """
    rng = np.random.default_rng(seed)
    arms = [f"arm_{i}" for i in range(n_arms)]
    contexts = rng.standard_normal((n_contexts, n_features))

    arm_weights: dict[str, np.ndarray] = {arm: rng.standard_normal(n_features) for arm in arms}
    # Make arm_0 the best (all positive, dominated by constant 0.8)
    arm_weights[arms[0]] = np.full(n_features, 0.8)

    def true_reward(arm: str, ctx: np.ndarray) -> float:
        linear = float(ctx @ arm_weights[arm])
        reward = 1.0 / (1.0 + np.exp(-linear))
        noise = rng.normal(0, 0.05)
        return float(np.clip(reward + noise, 0, 1))

    return contexts, arms, true_reward


def generate_nonlinear_synthetic(
    n_contexts: int = 500,
    n_features: int = 8,
    n_arms: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, list[str], Callable[[str, np.ndarray], float]]:
    """Generate non-linear reward data (sigmoid of quadratic interaction).

    Returns ``(contexts, arms, true_reward_fn)``.
    """
    rng = np.random.default_rng(seed)
    arms = [f"arm_{i}" for i in range(n_arms)]
    contexts = rng.standard_normal((n_contexts, n_features))

    arm_weights: dict[str, np.ndarray] = {arm: rng.standard_normal(n_features) for arm in arms}

    def true_reward(arm: str, ctx: np.ndarray) -> float:
        w = arm_weights[arm]
        linear = float(ctx @ w)
        reward = 1.0 / (1.0 + np.exp(-(linear + 0.3 * linear**2)))
        noise = rng.normal(0, 0.03)
        return float(np.clip(reward + noise, 0, 1))

    return contexts, arms, true_reward


def generate_biased_log_data(
    n_logs: int = 500,
    n_features: int = 5,
    arms: list[str] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate logged data with a biased logging policy.

    The logging policy favours the last arm (70% probability).
    The truly best arm is the *first* arm (mean reward 0.75).

    Returns ``(contexts, decisions, propensities, rewards)``.
    """
    rng = np.random.default_rng(seed)
    if arms is None:
        arms = ["arm_0", "arm_1", "arm_2"]
    n_arms = len(arms)

    contexts = rng.standard_normal((n_logs, n_features))

    # Logging policy: last arm 70%, rest split evenly
    high_prob = 0.70
    low_prob = (1.0 - high_prob) / (n_arms - 1)
    probs = [low_prob] * (n_arms - 1) + [high_prob]
    decisions = rng.choice(arms, size=n_logs, p=probs)
    propensities = np.array([probs[arms.index(a)] for a in decisions])

    # True reward: first arm is best
    true_means: dict[str, float] = {arms[0]: 0.75}
    for arm in arms[1:]:
        true_means[arm] = 0.3

    rewards = np.array([float(np.clip(rng.normal(true_means[a], 0.1), 0, 1)) for a in decisions])

    return contexts, decisions, propensities, rewards


# ---------------------------------------------------------------------------
# Part B — BanditSimulator
# ---------------------------------------------------------------------------


class BanditSimulator:
    """Wrap ``ClusterBandit`` in a streaming decide->update generator.

    Yields one ``StepRecord`` per decision for live charting.
    """

    def __init__(
        self,
        arms: list[str],
        n_features: int,
        policy: str | PolicyType = PolicyType.LIN_UCB,
        n_clusters: int = 3,
        seed: int = 42,
        true_reward_fn: Callable[[str, np.ndarray], float] | None = None,
        **bandit_kwargs: Any,
    ) -> None:
        self.arms = arms
        self.n_features = n_features
        self.policy = PolicyType(policy) if isinstance(policy, str) else policy
        self.n_clusters = n_clusters
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._true_reward_fn = true_reward_fn

        self._bandit: ClusterBandit | None = None
        self._bandit_kwargs = bandit_kwargs

        # Accumulators
        self._total_reward = 0.0
        self._total_optimal = 0.0
        self._step_count = 0

        # Full history (for replay / chart rebuild)
        self.history: list[StepRecord] = []

    # ---- bootstrap ----

    def bootstrap(self, contexts: np.ndarray, seed: int | None = None, **fit_kwargs: Any) -> None:
        """Fit the bandit on initial logged data so ``is_fitted`` is ``True``."""
        rng = np.random.default_rng(seed or self.seed)
        decisions = rng.choice(self.arms, size=len(contexts))
        rewards = np.clip(rng.normal(0.5, 0.2, size=len(contexts)), 0, 1)

        self._bandit = ClusterBandit(
            arms=self.arms,
            n_features=self.n_features,
            policy=self.policy,
            n_clusters=self.n_clusters,
            seed=self.seed,
            **self._bandit_kwargs,
        )
        self._bandit.fit_offline(contexts, decisions, rewards, **fit_kwargs)

    # ---- run ----

    def run(
        self,
        max_steps: int,
        contexts: np.ndarray,
        true_reward_fn: Callable[[str, np.ndarray], float] | None = None,
        sleep_s: float = 0.0,
    ) -> Generator[StepRecord, None, None]:
        """Streaming generator yielding one ``StepRecord`` per decision.

        Args:
            max_steps: Number of steps to run.
            contexts: Pre-generated context array to draw from (cycles if needed).
            true_reward_fn: Computes actual reward (also used for regret).
            sleep_s: Seconds to sleep between steps (for animation pacing).
        """
        if self._bandit is None:
            raise RuntimeError("Call bootstrap() before run()")

        effective_fn = true_reward_fn or self._true_reward_fn

        for i in range(max_steps):
            ctx = contexts[i % len(contexts)]
            decision = self._bandit.decide(ctx)
            chosen_arm = decision.chosen_arm

            # Compute reward
            if effective_fn is not None:
                reward = effective_fn(chosen_arm, ctx)
            else:
                reward = float(np.clip(self._rng.normal(0.5, 0.15), 0, 1))

            # Compute optimal reward (for regret)
            if effective_fn is not None:
                optimal_reward = max(effective_fn(a, ctx) for a in self.arms)
            else:
                optimal_reward = 1.0

            # Update accumulators
            self._step_count += 1
            self._total_reward += reward
            self._total_optimal += optimal_reward

            # Update bandit
            self._bandit.update(context=ctx, arm=chosen_arm, reward=reward)

            # Cluster assignment
            try:
                cluster = self._bandit.get_cluster_assignment(ctx)
            except Exception:
                cluster = -1

            record = StepRecord(
                step=self._step_count,
                context=ctx,
                chosen_arm=chosen_arm,
                reward=reward,
                cumulative_mean_reward=self._total_reward / self._step_count,
                cumulative_regret=self._total_optimal - self._total_reward,
                all_scores=decision.all_scores,
                cluster=cluster,
            )
            self.history.append(record)

            if sleep_s > 0:
                time.sleep(sleep_s)

            yield record

    # ---- convenience properties ----

    @property
    def bandit(self) -> ClusterBandit:
        if self._bandit is None:
            raise RuntimeError("Bandit not initialised. Call bootstrap() first.")
        return self._bandit

    @property
    def cumulative_mean_reward(self) -> float:
        return self._total_reward / max(1, self._step_count)

    @property
    def cumulative_regret(self) -> float:
        return self._total_optimal - self._total_reward


# ---------------------------------------------------------------------------
# Part C — Plotly chart factory functions
# ---------------------------------------------------------------------------

DEFAULT_COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]


def create_reward_chart(title: str = "Cumulative Mean Reward", n_arms: int = 3) -> go.Figure:
    """Create an empty live-updating cumulative reward chart."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            name="Cumulative Mean Reward",
            line=dict(color=DEFAULT_COLORS[0], width=2),
        ),
        secondary_y=False,
    )
    fig.add_hline(
        y=1.0 / n_arms,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Random baseline (1/{n_arms})",
    )
    fig.update_layout(
        title=title,
        xaxis_title="Step",
        template="plotly_white",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Mean Reward", range=[0, 1], secondary_y=False)
    return fig


def create_regret_chart(title: str = "Cumulative Regret") -> go.Figure:
    """Create an empty live-updating regret chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="lines",
            name="Cumulative Regret",
            line=dict(color=DEFAULT_COLORS[1], width=2),
            fill="tozeroy",
            fillcolor="rgba(239,85,59,0.1)",
        ),
    )
    fig.update_layout(
        title=title,
        xaxis_title="Step",
        yaxis_title="Regret",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def create_arm_score_chart(arms: list[str]) -> go.Figure:
    """Create a live-updating horizontal bar chart of per-arm scores."""
    colors = [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(len(arms))]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=list(arms),
            x=[0] * len(arms),
            orientation="h",
            marker_color=colors,
            text=[0] * len(arms),
            textposition="outside",
        ),
    )
    fig.update_layout(
        title="Current Arm Scores",
        xaxis_title="Score",
        xaxis_range=[0, 1.2],
        template="plotly_white",
        showlegend=False,
    )
    return fig


def create_cluster_distribution_chart(n_clusters: int = 3) -> go.Figure:
    """Create a live-updating donut/bar chart of cluster assignments."""
    fig = go.Figure()
    fig.add_trace(
        go.Pie(
            labels=[f"Cluster {i}" for i in range(n_clusters)],
            values=[0] * n_clusters,
            hole=0.4,
        ),
    )
    fig.update_layout(
        title="Context Cluster Distribution",
        template="plotly_white",
    )
    return fig


def create_arm_pull_chart(arms: list[str]) -> go.Figure:
    """Create a live-updating vertical bar chart of per-arm pull counts."""
    colors = [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(len(arms))]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(arms),
            y=[0] * len(arms),
            marker_color=colors,
        ),
    )
    fig.update_layout(
        title="Arm Pull Distribution",
        xaxis_title="Arm",
        yaxis_title="Times Chosen",
        template="plotly_white",
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Chart update helpers
# ---------------------------------------------------------------------------


def update_reward_chart(fig: go.Figure, steps: list[int], rewards: list[float]) -> go.Figure:
    """Update the cumulative reward trace with new data."""
    fig.data[0].x = steps
    fig.data[0].y = rewards
    return fig


def update_regret_chart(fig: go.Figure, steps: list[int], regrets: list[float]) -> go.Figure:
    """Update the cumulative regret trace with new data."""
    fig.data[0].x = steps
    fig.data[0].y = regrets
    return fig


def update_arm_score_chart(fig: go.Figure, scores: dict[Any, float]) -> go.Figure:
    """Update the arm score bar chart with latest scores."""
    arm_order = list(scores.keys())
    values = [scores[a] for a in arm_order]
    fig.data[0].y = arm_order
    fig.data[0].x = values
    fig.data[0].text = [f"{v:.3f}" for v in values]
    return fig


def update_cluster_chart(fig: go.Figure, cluster_counts: dict[int, int]) -> go.Figure:
    """Update the cluster distribution chart."""
    sorted_keys = sorted(cluster_counts)
    fig.data[0].labels = [f"Cluster {k}" for k in sorted_keys]
    fig.data[0].values = [cluster_counts[k] for k in sorted_keys]
    return fig


def update_arm_pull_chart(fig: go.Figure, pull_counts: dict[Any, int]) -> go.Figure:
    """Update the arm pull distribution bar chart."""
    arms_order = list(pull_counts.keys())
    fig.data[0].x = arms_order
    fig.data[0].y = [pull_counts[a] for a in arms_order]
    return fig
