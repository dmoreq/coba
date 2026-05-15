"""Tests for BinaryActionTree and ActionLeaf."""

import pytest

from coba.continuous.action_tree import ActionLeaf, BinaryActionTree


class TestActionLeaf:
    """Test ActionLeaf dataclass."""

    def test_valid_leaf(self) -> None:
        """ActionLeaf accepts valid bounds and computes midpoint."""
        leaf = ActionLeaf(index=0, lo=0.0, hi=1.0, midpoint=0.5)
        assert leaf.index == 0
        assert leaf.lo == 0.0
        assert leaf.hi == 1.0
        assert leaf.midpoint == 0.5

    def test_leaf_bounds_invalid(self) -> None:
        """leaf with lo >= hi raises ValueError."""
        with pytest.raises(ValueError, match="bounds invalid"):
            ActionLeaf(index=0, lo=1.0, hi=1.0, midpoint=1.0)

    def test_leaf_bounds_reversed(self) -> None:
        """leaf with lo > hi raises ValueError."""
        with pytest.raises(ValueError, match="bounds invalid"):
            ActionLeaf(index=0, lo=1.0, hi=0.0, midpoint=0.5)

    def test_leaf_midpoint_incorrect(self) -> None:
        """leaf with incorrect midpoint raises ValueError."""
        with pytest.raises(ValueError, match="midpoint .* does not match"):
            ActionLeaf(index=0, lo=0.0, hi=1.0, midpoint=0.3)

    def test_leaf_contains_action(self) -> None:
        """ActionLeaf.contains() checks interval membership."""
        leaf = ActionLeaf(index=0, lo=1.0, hi=2.0, midpoint=1.5)
        assert leaf.contains(1.0)
        assert leaf.contains(1.5)
        assert leaf.contains(2.0)
        assert not leaf.contains(0.9)
        assert not leaf.contains(2.1)

    def test_leaf_clamp_action(self) -> None:
        """ActionLeaf.clamp() clamps action to bounds."""
        leaf = ActionLeaf(index=0, lo=1.0, hi=2.0, midpoint=1.5)
        assert leaf.clamp(0.5) == 1.0
        assert leaf.clamp(1.5) == 1.5
        assert leaf.clamp(2.5) == 2.0

    def test_leaf_frozen(self) -> None:
        """ActionLeaf is immutable (frozen=True)."""
        leaf = ActionLeaf(index=0, lo=0.0, hi=1.0, midpoint=0.5)
        with pytest.raises(AttributeError):
            leaf.index = 1  # type: ignore[misc]


class TestBinaryActionTree:
    """Test BinaryActionTree construction and operations."""

    def test_tree_creation_default_depth(self) -> None:
        """Tree creates with default depth=6 (64 leaves)."""
        tree = BinaryActionTree(a_min=0.0, a_max=1.0)
        assert tree.depth == 6
        assert tree.n_leaves == 64
        assert tree.a_min == 0.0
        assert tree.a_max == 1.0

    def test_tree_custom_depth(self) -> None:
        """Tree with custom depth creates correct n_leaves."""
        for depth in [1, 2, 4, 8]:
            tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=depth)
            assert tree.n_leaves == 2**depth
            assert tree.depth == depth

    def test_tree_invalid_bounds(self) -> None:
        """Tree with a_min >= a_max raises ValueError."""
        with pytest.raises(ValueError, match="invalid action bounds"):
            BinaryActionTree(a_min=1.0, a_max=1.0)
        with pytest.raises(ValueError, match="invalid action bounds"):
            BinaryActionTree(a_min=1.0, a_max=0.0)

    def test_tree_invalid_depth_zero(self) -> None:
        """Tree with depth < 1 raises ValueError."""
        with pytest.raises(ValueError, match="depth must be at least 1"):
            BinaryActionTree(a_min=0.0, a_max=1.0, depth=0)

    def test_tree_invalid_depth_too_large(self) -> None:
        """Tree with depth > 16 raises ValueError."""
        with pytest.raises(ValueError, match="depth > 16"):
            BinaryActionTree(a_min=0.0, a_max=1.0, depth=17)

    def test_tree_bandwidth_calculation(self) -> None:
        """Bandwidth is half of leaf_width."""
        tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=3)  # 8 leaves
        expected_leaf_width = 1.0 / 8
        expected_bandwidth = expected_leaf_width / 2.0
        assert abs(tree.leaf_width - expected_leaf_width) < 1e-10
        assert abs(tree.bandwidth - expected_bandwidth) < 1e-10

    def test_tree_leaf_width_varies_with_bounds(self) -> None:
        """Leaf width scales with action space size."""
        tree1 = BinaryActionTree(a_min=0.0, a_max=1.0, depth=3)
        tree2 = BinaryActionTree(a_min=0.0, a_max=10.0, depth=3)
        assert abs(tree2.leaf_width - 10 * tree1.leaf_width) < 1e-10

    def test_tree_leaves_list(self) -> None:
        """Tree.leaves() returns all leaves in order."""
        tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=2)  # 4 leaves
        leaves = tree.leaves()
        assert len(leaves) == 4
        assert all(isinstance(leaf, ActionLeaf) for leaf in leaves)
        # Check indices are 0, 1, 2, 3
        assert [leaf.index for leaf in leaves] == [0, 1, 2, 3]
        # Check bounds are in order
        for i, leaf in enumerate(leaves):
            assert leaf.lo == i * tree.leaf_width
            assert leaf.hi == (i + 1) * tree.leaf_width

    def test_leaf_for_action_interior(self) -> None:
        """leaf_for_action() assigns interior actions to correct leaf."""
        tree = BinaryActionTree(
            a_min=0.0, a_max=1.0, depth=2
        )  # 4 leaves: [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0]
        # Action 0.1 should be in leaf 0
        leaf = tree.leaf_for_action(0.1)
        assert leaf.index == 0
        assert leaf.contains(0.1)
        # Action 0.6 should be in leaf 2
        leaf = tree.leaf_for_action(0.6)
        assert leaf.index == 2
        assert leaf.contains(0.6)

    def test_leaf_for_action_boundaries(self) -> None:
        """leaf_for_action() correctly handles leaf boundaries."""
        tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=2)
        # Exact left boundary of leaf 0
        leaf = tree.leaf_for_action(0.0)
        assert leaf.index == 0
        # Exact right boundary of leaf 0
        leaf = tree.leaf_for_action(0.25)
        assert leaf.index == 1  # right boundary belongs to next leaf
        # Exact right boundary of tree
        leaf = tree.leaf_for_action(1.0)
        assert leaf.index == 3

    def test_leaf_for_action_clamping_below(self) -> None:
        """leaf_for_action() clamps action below a_min to first leaf."""
        tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=2)
        leaf = tree.leaf_for_action(-0.5)
        assert leaf.index == 0
        assert leaf.contains(0.0)

    def test_leaf_for_action_clamping_above(self) -> None:
        """leaf_for_action() clamps action above a_max to last leaf."""
        tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=2)
        leaf = tree.leaf_for_action(1.5)
        assert leaf.index == 3
        assert leaf.contains(1.0)

    def test_tree_with_negative_bounds(self) -> None:
        """Tree works with negative action bounds."""
        tree = BinaryActionTree(a_min=-5.0, a_max=5.0, depth=3)
        assert tree.a_min == -5.0
        assert tree.a_max == 5.0
        assert tree.n_leaves == 8
        # Action 0.0 should be roughly in the middle
        leaf = tree.leaf_for_action(0.0)
        assert leaf.contains(0.0)

    def test_tree_with_float_precision(self) -> None:
        """Tree handles floating-point precision edge cases."""
        tree = BinaryActionTree(a_min=0.5, a_max=4.5, depth=4)
        # Check that all leaves partition the space without gaps
        for i, leaf in enumerate(tree.leaves()):
            if i > 0:
                prev_leaf = tree.leaves()[i - 1]
                assert abs(leaf.lo - prev_leaf.hi) < 1e-10  # adjacent leaves

    def test_tree_leaf_count_consistency(self) -> None:
        """All leaves returned by leaves() match n_leaves."""
        for depth in [1, 2, 4, 6, 8]:
            tree = BinaryActionTree(a_min=0.0, a_max=1.0, depth=depth)
            assert len(tree.leaves()) == tree.n_leaves

    def test_tree_repr(self) -> None:
        """Tree has informative repr."""
        tree = BinaryActionTree(a_min=0.5, a_max=5.0, depth=4)
        repr_str = repr(tree)
        assert "BinaryActionTree" in repr_str
        assert "0.5" in repr_str
        assert "5.0" in repr_str
        assert "4" in repr_str
        assert "16" in repr_str  # 2^4
