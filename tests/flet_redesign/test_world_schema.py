"""Tests for world schema validation."""

from __future__ import annotations

import pytest

from coba.flet_redesign.worlds.schema import ArmDef, FeatureDef, WorldConfig


def test_feature_def_validates_numeric_range() -> None:
    with pytest.raises(ValueError, match="numeric_min"):
        FeatureDef(name="x", feature_type="numeric", numeric_min=1.0, numeric_max=1.0)


def test_feature_def_validates_categorical_categories() -> None:
    with pytest.raises(ValueError, match="at least two categories"):
        FeatureDef(name="segment", feature_type="categorical", categories=("a",))


def test_arm_def_validates_base_rate() -> None:
    with pytest.raises(ValueError, match="base_rate"):
        ArmDef(arm_id="a", label="A", base_rate=1.2)


def test_world_config_requires_unique_features_and_arms() -> None:
    f1 = FeatureDef(name="x", feature_type="numeric", numeric_min=0.0, numeric_max=1.0)
    f2 = FeatureDef(name="x", feature_type="numeric", numeric_min=0.0, numeric_max=1.0)
    arm_a = ArmDef(arm_id="a", label="A", base_rate=0.5, weights={"x": 0.1})
    arm_b = ArmDef(arm_id="b", label="B", base_rate=0.4, weights={"x": 0.1})
    with pytest.raises(ValueError, match="feature names must be unique"):
        WorldConfig(
            world_id="w",
            title="W",
            description="desc",
            difficulty="easy",
            features=(f1, f2),
            arms=(arm_a, arm_b),
        )


def test_world_config_rejects_unknown_arm_weight_feature() -> None:
    f1 = FeatureDef(name="x", feature_type="numeric", numeric_min=0.0, numeric_max=1.0)
    arm_a = ArmDef(arm_id="a", label="A", base_rate=0.5, weights={"y": 0.1})
    arm_b = ArmDef(arm_id="b", label="B", base_rate=0.4, weights={"x": 0.1})
    with pytest.raises(ValueError, match="unknown feature weights"):
        WorldConfig(
            world_id="w",
            title="W",
            description="desc",
            difficulty="easy",
            features=(f1,),
            arms=(arm_a, arm_b),
        )
