"""Binary tree partitioning of continuous action space.

A CATS action tree recursively partitions [a_min, a_max] into 2^depth
equal-width leaves. Each leaf has a LinUCB model and a bandwidth window
for smoothed action sampling.

Design rationale:
  - Equal-width leaves → predictable tree structure, simple leaf assignment
  - Binary tree depth configurable → scales from coarse (4 leaves) to fine (256 leaves)
  - Leaf bandwidth h = half the leaf width → used for exploration sampling window
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionLeaf:
    """One leaf (bucket) of the CATS action tree.

    Attributes:
        index: Leaf index in [0, n_leaves-1].
        lo: Lower bound of this leaf's action interval.
        hi: Upper bound of this leaf's action interval.
        midpoint: Center of the interval, (lo + hi) / 2.
    """

    index: int
    lo: float
    hi: float
    midpoint: float

    def __post_init__(self) -> None:
        """Validate leaf bounds."""
        if not (self.lo < self.hi):
            raise ValueError(f"leaf bounds invalid: lo={self.lo} >= hi={self.hi}")
        expected_mid = (self.lo + self.hi) / 2
        if abs(self.midpoint - expected_mid) > 1e-10:
            raise ValueError(f"midpoint {self.midpoint} does not match (lo+hi)/2 = {expected_mid}")

    def contains(self, action: float) -> bool:
        """Check if action falls within this leaf's interval."""
        return self.lo <= action <= self.hi

    def clamp(self, action: float) -> float:
        """Clamp action to this leaf's bounds."""
        return max(self.lo, min(self.hi, action))


class BinaryActionTree:
    """Partitions [a_min, a_max] into 2^depth equal-width leaves.

    Each leaf owns a LinUCB bandit model. At decision time, all leaves are
    scored, and the action is sampled uniformly within the best leaf's
    ±bandwidth window.

    Args:
        a_min: Lower bound of action space.
        a_max: Upper bound of action space.
        depth: Tree depth. n_leaves = 2^depth. Typical: 4–8.

    Attributes:
        n_leaves: Number of leaves = 2^depth.
        bandwidth: Half the width of one leaf = (a_max - a_min) / (2 * n_leaves).
    """

    def __init__(self, a_min: float, a_max: float, depth: int = 6) -> None:
        if not (a_min < a_max):
            raise ValueError(f"invalid action bounds: a_min={a_min} >= a_max={a_max}")
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        if depth > 16:
            raise ValueError(f"depth > 16 (2^16 = 65536 leaves) is impractical, got {depth}")

        self.a_min = float(a_min)
        self.a_max = float(a_max)
        self.depth = int(depth)
        self._n_leaves = 2**depth
        self._leaf_width = (self.a_max - self.a_min) / self._n_leaves
        self._bandwidth = self._leaf_width / 2.0

        # Pre-compute all leaves
        self._leaves: list[ActionLeaf] = []
        for i in range(self._n_leaves):
            lo = self.a_min + i * self._leaf_width
            hi = lo + self._leaf_width
            mid = (lo + hi) / 2.0
            self._leaves.append(ActionLeaf(index=i, lo=lo, hi=hi, midpoint=mid))

    @property
    def n_leaves(self) -> int:
        """Number of leaves in the tree."""
        return self._n_leaves

    @property
    def leaf_width(self) -> float:
        """Width of one leaf = (a_max - a_min) / n_leaves."""
        return self._leaf_width

    @property
    def bandwidth(self) -> float:
        """Bandwidth for smoothed sampling = leaf_width / 2."""
        return self._bandwidth

    def leaf_for_action(self, action: float) -> ActionLeaf:
        """Find the leaf containing the given action.

        If action is outside [a_min, a_max], clamps to the nearest boundary leaf.

        Args:
            action: A scalar action value.

        Returns:
            ActionLeaf containing or clamping to this action.
        """
        clamped = max(self.a_min, min(self.a_max, action))
        # Map clamped action to leaf index: idx = floor((clamped - a_min) / leaf_width)
        idx = int((clamped - self.a_min) / self._leaf_width)
        # Clamp index to valid range in case of floating-point boundary issues
        idx = max(0, min(self._n_leaves - 1, idx))
        return self._leaves[idx]

    def leaves(self) -> list[ActionLeaf]:
        """Return all leaves in order."""
        return list(self._leaves)

    def __repr__(self) -> str:
        return (
            f"BinaryActionTree(a_min={self.a_min}, a_max={self.a_max}, "
            f"depth={self.depth}, n_leaves={self.n_leaves})"
        )
