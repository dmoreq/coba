"""Tests for Flet redesign contract definitions."""

from __future__ import annotations

from typing import Any

from coba.flet_redesign.contracts import BanditPolicy, DebugSnapshotProvider, World


class DummyPolicy:
    def reset(self) -> None:
        return None

    def select_arm(self, context: dict[str, float], arms: list[str]) -> str:
        return arms[0]

    def update(self, context: dict[str, float], arm: str, reward: float) -> None:
        return None


class DummyWorld:
    def reset(self, seed: int | None = None) -> None:
        _ = seed

    def get_available_arms(self) -> list[str]:
        return ["a", "b"]

    def sample_context(self, step_index: int) -> dict[str, Any]:
        return {"step": step_index}

    def sample_reward(self, context: dict[str, Any], arm: str) -> float:
        _ = context, arm
        return 1.0


class DummyDebugger:
    def get_debug_snapshot(self) -> dict[str, Any]:
        return {"ok": True}


def test_bandit_policy_protocol_runtime_check() -> None:
    assert isinstance(DummyPolicy(), BanditPolicy)


def test_world_protocol_runtime_check() -> None:
    assert isinstance(DummyWorld(), World)


def test_debug_snapshot_provider_runtime_check() -> None:
    assert isinstance(DummyDebugger(), DebugSnapshotProvider)
