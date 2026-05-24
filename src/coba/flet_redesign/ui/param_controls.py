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
    return ()
