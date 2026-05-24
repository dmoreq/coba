"""Tests for trace JSON/CSV export-import and filtering."""

from __future__ import annotations

import json

from web.contracts import SimulationStepResult
from web.trace import TraceBuffer, filter_trace_records


def _sample_buffer() -> TraceBuffer:
    buffer = TraceBuffer()
    for idx, arm in [(1, "a"), (2, "b"), (3, "a")]:
        buffer.append(
            SimulationStepResult(
                step_index=idx,
                context={"step": idx},
                chosen_arm=arm,
                reward=float(idx % 2),
                cumulative_reward=float(idx),
                cumulative_regret=float(3 - idx),
                metadata={"uncertainty": 0.1 * idx},
            )
        )
    return buffer


def test_trace_json_roundtrip() -> None:
    original = _sample_buffer()
    payload = original.to_json()
    restored = TraceBuffer.from_json(payload)
    assert restored.to_records() == original.to_records()


def test_trace_csv_export_has_expected_columns() -> None:
    csv_payload = _sample_buffer().to_csv()
    lines = [line for line in csv_payload.splitlines() if line.strip()]
    assert lines[0].startswith("step_index,context,chosen_arm,reward,cumulative_reward")
    assert len(lines) == 4


def test_filter_trace_records_by_token() -> None:
    records = _sample_buffer().to_records()
    filtered = filter_trace_records(records, "chosen_arm")
    assert len(filtered) == len(records)

    filtered_arm = filter_trace_records(records, '"b"')
    assert len(filtered_arm) == 1
    assert filtered_arm[0]["chosen_arm"] == "b"


def test_trace_from_records_accepts_serialized_payload() -> None:
    records = json.loads(_sample_buffer().to_json())
    rebuilt = TraceBuffer.from_records(records)
    assert len(rebuilt.steps) == 3
    assert rebuilt.steps[1].chosen_arm == "b"
