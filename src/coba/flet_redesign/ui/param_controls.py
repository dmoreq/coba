"""Parameter control specs and model-level validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from coba.flet_redesign.ui.tooltips import ParamTooltip

ControlType = Literal["slider", "toggle", "select"]


@dataclass(frozen=True)
class ParamControlSpec:
    """Specification for rendering one parameter control."""

    key: str
    label: str
    control_type: ControlType
    default_value: Any
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    options: tuple[str, ...] = ()
    tooltip: ParamTooltip | None = None

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("ParamControlSpec.key cannot be empty")
        if self.control_type == "slider":
            if self.min_value is None or self.max_value is None or self.step is None:
                raise ValueError("slider controls require min_value, max_value, and step")
            if self.min_value >= self.max_value:
                raise ValueError("slider min_value must be < max_value")
        if self.control_type == "select" and not self.options:
            raise ValueError("select controls require non-empty options")


def default_policy_param_controls(policy_id: str) -> tuple[ParamControlSpec, ...]:
    """Return default controls by policy id."""
    if policy_id == "epsilon_greedy":
        return (
            ParamControlSpec(
                key="epsilon",
                label="Epsilon",
                control_type="slider",
                default_value=0.1,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                tooltip=ParamTooltip(
                    title="Epsilon",
                    intuition="Higher epsilon increases random exploration.",
                    formula="P(explore) = epsilon",
                    tuning_hint="Start at 0.1 and reduce if reward variance is low.",
                ),
            ),
        )
    if policy_id == "ucb1":
        return (
            ParamControlSpec(
                key="alpha",
                label="Alpha",
                control_type="slider",
                default_value=1.0,
                min_value=0.1,
                max_value=3.0,
                step=0.1,
                tooltip=ParamTooltip(
                    title="UCB Alpha",
                    intuition="Alpha scales confidence width for exploration.",
                    formula="score = mean + alpha * sqrt(2 log t / n)",
                    tuning_hint="Increase alpha in non-stationary or sparse-reward setups.",
                ),
            ),
        )
    if policy_id == "softmax":
        return (
            ParamControlSpec(
                key="tau",
                label="Temperature",
                control_type="slider",
                default_value=0.2,
                min_value=0.05,
                max_value=1.0,
                step=0.05,
                tooltip=ParamTooltip(
                    title="Temperature",
                    intuition="Lower temperature makes action choice more greedy.",
                    formula="P(a) ∝ exp(Q(a) / tau)",
                    tuning_hint="Try 0.2 as baseline; increase for broader exploration.",
                ),
            ),
        )
    if policy_id == "linucb":
        return (
            ParamControlSpec(
                key="alpha",
                label="Alpha",
                control_type="slider",
                default_value=1.0,
                min_value=0.1,
                max_value=3.0,
                step=0.1,
                tooltip=ParamTooltip(
                    title="LinUCB Alpha",
                    intuition="Scales contextual exploration bonus.",
                    formula="score = theta^T x + alpha * sqrt(x^T A^-1 x)",
                    tuning_hint="Increase alpha for sparse contextual coverage.",
                ),
            ),
            ParamControlSpec(
                key="l2_lambda",
                label="L2 Lambda",
                control_type="slider",
                default_value=1.0,
                min_value=0.1,
                max_value=5.0,
                step=0.1,
                tooltip=ParamTooltip(
                    title="L2 Regularization",
                    intuition="Stabilizes early updates in high-dimensional contexts.",
                    formula="A_0 = lambda * I",
                    tuning_hint="Increase when features are noisy or collinear.",
                ),
            ),
        )
    if policy_id == "linucb_sw":
        return (
            ParamControlSpec(
                key="window_size",
                label="Window Size",
                control_type="slider",
                default_value=200,
                min_value=20,
                max_value=500,
                step=10,
                tooltip=ParamTooltip(
                    title="Sliding Window",
                    intuition="Smaller windows adapt faster to drift.",
                    formula="fit on last W updates",
                    tuning_hint="Reduce W when reward dynamics change quickly.",
                ),
            ),
        )
    if policy_id == "logistic_ucb":
        return (
            ParamControlSpec(
                key="learning_rate",
                label="Learning Rate",
                control_type="slider",
                default_value=0.1,
                min_value=0.01,
                max_value=0.5,
                step=0.01,
                tooltip=ParamTooltip(
                    title="Learning Rate",
                    intuition="Controls step size in logistic updates.",
                    formula="theta <- theta + eta * gradient",
                    tuning_hint="Lower eta when updates oscillate.",
                ),
            ),
            ParamControlSpec(
                key="alpha",
                label="Alpha",
                control_type="slider",
                default_value=0.5,
                min_value=0.1,
                max_value=2.0,
                step=0.1,
                tooltip=ParamTooltip(
                    title="Exploration Bonus",
                    intuition="Compensates low-confidence contextual regions.",
                    formula="score = p_hat + alpha / sqrt(n+1)",
                    tuning_hint="Increase alpha if model over-exploits early.",
                ),
            ),
        )
    if policy_id == "cats":
        return (
            ParamControlSpec(
                key="action_min",
                label="Action Min",
                control_type="slider",
                default_value=0.0,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                tooltip=ParamTooltip(
                    title="Action Lower Bound",
                    intuition="Lower bound for continuous action sampling.",
                    formula="a_t ∈ [min, max]",
                    tuning_hint="Use domain-safe lower bounds for production runs.",
                ),
            ),
            ParamControlSpec(
                key="action_max",
                label="Action Max",
                control_type="slider",
                default_value=1.0,
                min_value=0.1,
                max_value=2.0,
                step=0.01,
                tooltip=ParamTooltip(
                    title="Action Upper Bound",
                    intuition="Upper bound for continuous action sampling.",
                    formula="a_t ∈ [min, max]",
                    tuning_hint="Keep range narrow when reward landscape is smooth.",
                ),
            ),
            ParamControlSpec(
                key="exploration",
                label="Exploration",
                control_type="slider",
                default_value=0.25,
                min_value=0.05,
                max_value=1.0,
                step=0.05,
                tooltip=ParamTooltip(
                    title="CATS Exploration",
                    intuition="Controls sampling spread around the current best action.",
                    formula="a_t ~ N(best, exploration * range)",
                    tuning_hint="Increase exploration in non-stationary settings.",
                ),
            ),
        )
    return ()
