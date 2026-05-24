"""Tests for drift timeline, checkpointing, and preset management."""

from __future__ import annotations

from pathlib import Path

from coba.flet_redesign.checkpoint import (
    CheckpointPayload,
    load_checkpoint,
    save_checkpoint,
)
from coba.flet_redesign.drift_monitor import DriftTimeline
from coba.flet_redesign.preset_manager import Preset, PresetManager


def test_drift_timeline_records_events() -> None:
    timeline = DriftTimeline(delta=0.0, lambda_=0.2, alpha=0.8)
    values = [0.1, 0.1, 0.1, 0.9, 0.95, 1.0]
    detected = [
        timeline.update(step_index=index + 1, value=value) for index, value in enumerate(values)
    ]
    assert any(detected)
    assert timeline.indicators()["num_events"] >= 1


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    payload = CheckpointPayload(
        checkpoint_id="ckpt-001",
        kind="discrete",
        state={"step": 10},
        trace=[{"step_index": 1, "reward": 1.0}],
    )
    save_checkpoint(path, payload)
    loaded = load_checkpoint(path)
    assert loaded == payload


def test_preset_manager_roundtrip(tmp_path: Path) -> None:
    manager = PresetManager(path=tmp_path / "presets.json")
    presets = [
        Preset(
            preset_id="p1",
            title="Continuous Baseline",
            payload={"policy_id": "cats", "world_id": "ridepilot"},
        ),
    ]
    manager.save_presets(presets)
    loaded = manager.load_presets()
    assert loaded == presets
