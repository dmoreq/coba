"""Checkpoint persistence helpers for discrete/continuous runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from coba.flet_redesign.continuous.schemas import ContinuousStepResult
from coba.flet_redesign.contracts import SimulationStepResult


@dataclass(frozen=True)
class CheckpointPayload:
    """Serialized checkpoint payload."""

    checkpoint_id: str
    kind: str
    state: dict[str, Any]
    trace: list[dict[str, Any]]


def save_checkpoint(path: Path, payload: CheckpointPayload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(payload), ensure_ascii=True, indent=2), encoding="utf-8")


def load_checkpoint(path: Path) -> CheckpointPayload:
    data = json.loads(path.read_text(encoding="utf-8"))
    return CheckpointPayload(
        checkpoint_id=str(data["checkpoint_id"]),
        kind=str(data["kind"]),
        state=dict(data["state"]),
        trace=list(data["trace"]),
    )


def discrete_trace_to_records(trace: list[SimulationStepResult]) -> list[dict[str, Any]]:
    return [
        {
            "step_index": step.step_index,
            "context": step.context,
            "chosen_arm": step.chosen_arm,
            "reward": step.reward,
            "cumulative_reward": step.cumulative_reward,
            "cumulative_regret": step.cumulative_regret,
            "metadata": step.metadata,
        }
        for step in trace
    ]


def continuous_trace_to_records(trace: list[ContinuousStepResult]) -> list[dict[str, Any]]:
    return [
        {
            "step_index": step.step_index,
            "context": step.context,
            "action": step.action,
            "reward": step.reward,
            "cumulative_reward": step.cumulative_reward,
            "metadata": step.metadata,
        }
        for step in trace
    ]
