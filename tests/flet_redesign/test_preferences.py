"""Tests for local preference persistence."""

from __future__ import annotations

from pathlib import Path

from coba.flet_redesign.ui.preferences import PreferencesStore, UserPreferences


def test_preferences_store_returns_defaults_when_missing(tmp_path: Path) -> None:
    store = PreferencesStore(file_path=tmp_path / "prefs.json")
    prefs = store.load()
    assert prefs.world_id == "rural_clinic"
    assert prefs.policy_id == "random"
    assert prefs.speed == "1x"


def test_preferences_store_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "prefs.json"
    store = PreferencesStore(file_path=target)
    original = UserPreferences(world_id="newsfeed", policy_id="ucb1", speed="10x")
    store.save(original)

    loaded = store.load()
    assert loaded == original
