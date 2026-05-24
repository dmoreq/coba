"""Tests for trace buffering and export."""

from __future__ import annotations

import json

from coba.flet_redesign.contracts import SimulationStepResult
from coba.flet_redesign.trace import TraceBuffer


def test_trace_buffer_collects_steps() -> None:
    buffer = TraceBuffer()
    buffer.append(
        SimulationStepResult(
            step_index=1,
            context={"feature": 1},
            chosen_arm="arm_a",
            reward=0.8,
            cumulative_reward=0.8,
            cumulative_regret=0.2,
        )
    )
    assert len(buffer) == 1
    assert buffer.steps[0].chosen_arm == "arm_a"


def test_trace_buffer_json_serialization_handles_non_json_objects() -> None:
    buffer = TraceBuffer()
    buffer.append(
        SimulationStepResult(
            step_index=1,
            context={1, 2, 3},
            chosen_arm=("arm", 1),
            reward=1.0,
            cumulative_reward=1.0,
            cumulative_regret=0.0,
            metadata={"debug": {"a", "b"}},
        )
    )
    payload = json.loads(buffer.to_json())
    assert payload[0]["context"] == "{1, 2, 3}"
    assert payload[0]["chosen_arm"] == ["arm", 1]
    assert "{'a', 'b'}" == payload[0]["metadata"]["debug"]
