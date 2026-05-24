"""Tests for registered core worlds and world switching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from web.state_store import AppStateStore
from web.worlds import create_world, get_world_config, list_world_configs


def _fixture_payload() -> dict[str, Any]:
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "core_world_fixtures.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def test_registered_worlds_match_fixture_metadata() -> None:
    payload = _fixture_payload()
    fixture_worlds = cast(list[dict[str, int | str]], payload["worlds"])
    expected = {item["world_id"]: (item["n_features"], item["n_arms"]) for item in fixture_worlds}

    configs = list_world_configs()
    assert {config.world_id for config in configs} == set(expected.keys())
    for config in configs:
        n_features, n_arms = expected[config.world_id]
        assert len(config.features) == n_features
        assert len(config.arms) == n_arms


@pytest.mark.parametrize(
    "world_id",
    ["rural_clinic", "moviematch", "newsfeed", "shopsmart", "ridepilot", "gamebot", "labtrial"],
)
def test_world_sampling_is_deterministic_for_same_seed(world_id: str) -> None:
    world_a = create_world(world_id)
    world_b = create_world(world_id)
    world_a.reset(seed=42)
    world_b.reset(seed=42)

    contexts_a = [world_a.sample_context(step_index=i) for i in range(1, 6)]
    contexts_b = [world_b.sample_context(step_index=i) for i in range(1, 6)]
    assert contexts_a == contexts_b

    rewards_a = [
        world_a.sample_reward(context=contexts_a[i], arm=world_a.get_available_arms()[0])
        for i in range(5)
    ]
    rewards_b = [
        world_b.sample_reward(context=contexts_b[i], arm=world_b.get_available_arms()[0])
        for i in range(5)
    ]
    assert rewards_a == rewards_b


def test_world_switching_updates_state_and_builds_world() -> None:
    store = AppStateStore(default_world_id="rural_clinic")
    assert store.state.world_id == "rural_clinic"

    next_state = store.switch_world("newsfeed")
    assert next_state.world_id == "newsfeed"

    world = store.build_world()
    assert world.config.world_id == "newsfeed"
    assert len(world.get_available_arms()) == 3


def test_unknown_world_lookup_raises_key_error() -> None:
    with pytest.raises(KeyError, match="Unknown world_id"):
        get_world_config("unknown")
