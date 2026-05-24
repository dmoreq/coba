"""Three-pane layout specs used by lesson/arena/sandbox routes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaneSpec:
    """One pane spec for desktop layout."""

    key: str
    title: str
    width_ratio: float


@dataclass(frozen=True)
class ThreePaneLayoutSpec:
    """Desktop layout with three stable panes."""

    left: PaneSpec
    center: PaneSpec
    right: PaneSpec

    @property
    def total_ratio(self) -> float:
        return self.left.width_ratio + self.center.width_ratio + self.right.width_ratio


def build_three_pane_layout() -> ThreePaneLayoutSpec:
    """Return the default redesign desktop layout.

    Ratios sum to 1.0 for stable split rendering:
    - left: scene/context
    - center: treatment/action
    - right: metrics/debugger
    """
    return ThreePaneLayoutSpec(
        left=PaneSpec(key="scene", title="Scene", width_ratio=0.25),
        center=PaneSpec(key="treatment", title="Treatment", width_ratio=0.5),
        right=PaneSpec(key="metrics", title="Metrics", width_ratio=0.25),
    )
