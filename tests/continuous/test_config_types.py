"""Tests for CATS-related config and type extensions."""

from coba.config import BanditConfig
from coba.types import PolicyType


class TestBanditConfigCATS:
    """Test CATS fields in BanditConfig."""

    def test_config_default_cats_values(self) -> None:
        """BanditConfig has sensible defaults for CATS parameters."""
        config = BanditConfig()
        assert config.cats_a_min == 0.0
        assert config.cats_a_max == 1.0
        assert config.cats_depth == 6

    def test_config_custom_cats_values(self) -> None:
        """BanditConfig accepts custom CATS parameters."""
        config = BanditConfig(
            cats_a_min=0.50,
            cats_a_max=5.00,
            cats_depth=8,
        )
        assert config.cats_a_min == 0.50
        assert config.cats_a_max == 5.00
        assert config.cats_depth == 8

    def test_config_cats_fields_are_numeric(self) -> None:
        """CATS fields support numeric operations."""
        config = BanditConfig(cats_a_min=1.0, cats_a_max=10.0, cats_depth=5)
        action_range = config.cats_a_max - config.cats_a_min
        assert action_range == 9.0
        n_leaves = 2**config.cats_depth
        assert n_leaves == 32

    def test_config_cats_independent_of_cluster_params(self) -> None:
        """CATS parameters don't affect cluster parameters."""
        config = BanditConfig(
            n_clusters=3,
            cats_depth=4,
        )
        assert config.n_clusters == 3
        assert config.cats_depth == 4
        assert 2**config.cats_depth == 16


class TestPolicyTypeCATS:
    """Test CATS PolicyType enum value."""

    def test_policy_type_cats_exists(self) -> None:
        """PolicyType.CATS is defined."""
        assert PolicyType.CATS is not None
        assert PolicyType.CATS.value == "cats"

    def test_policy_type_cats_is_string_enum(self) -> None:
        """PolicyType.CATS behaves like a string enum."""
        assert isinstance(PolicyType.CATS, str)
        assert PolicyType.CATS.value == "cats"

    def test_policy_type_cats_in_enum_list(self) -> None:
        """PolicyType.CATS is in the list of all PolicyType values."""
        all_policies = list(PolicyType)
        assert PolicyType.CATS in all_policies

    def test_policy_type_cats_can_be_constructed_from_string(self) -> None:
        """PolicyType can be constructed from 'cats' string."""
        policy = PolicyType("cats")
        assert policy == PolicyType.CATS
        assert policy.value == "cats"

    def test_policy_type_cats_distinct_from_others(self) -> None:
        """PolicyType.CATS is distinct from other policies."""
        assert PolicyType.CATS != PolicyType.LIN_UCB
        assert PolicyType.CATS != PolicyType.LIN_TS
        assert PolicyType.CATS != PolicyType.THOMPSON

    def test_all_policy_types_have_unique_values(self) -> None:
        """All PolicyType values are unique."""
        all_values = [p.value for p in PolicyType]
        assert len(all_values) == len(set(all_values))
