"""Trace utilities for simulation replay and export."""

from __future__ import annotations

import json
from csv import DictWriter
from dataclasses import asdict
from io import StringIO
from typing import Any

from coba.flet_redesign.contracts import SimulationStepResult


class TraceBuffer:
    """In-memory trace buffer with JSON/row export helpers."""

    def __init__(self) -> None:
        self._steps: list[SimulationStepResult] = []

    def __len__(self) -> int:
        return len(self._steps)

    @property
    def steps(self) -> list[SimulationStepResult]:
        return list(self._steps)

    def append(self, step: SimulationStepResult) -> None:
        self._steps.append(step)

    def clear(self) -> None:
        self._steps.clear()

    def to_records(self) -> list[dict[str, Any]]:
        """Return JSON-serializable dictionaries for all steps."""
        records: list[dict[str, Any]] = []
        for step in self._steps:
            payload = asdict(step)
            payload["context"] = _to_json_value(payload["context"])
            payload["chosen_arm"] = _to_json_value(payload["chosen_arm"])
            payload["metadata"] = _to_json_value(payload["metadata"])
            records.append(payload)
        return records

    def to_json(self) -> str:
        """Serialize trace as formatted JSON."""
        return json.dumps(self.to_records(), ensure_ascii=True, indent=2, sort_keys=True)

    def to_csv(self) -> str:
        """Serialize trace as CSV."""
        records = self.to_records()
        if not records:
            return ""
        fieldnames = [
            "step_index",
            "context",
            "chosen_arm",
            "reward",
            "cumulative_reward",
            "cumulative_regret",
            "metadata",
        ]
        stream = StringIO()
        writer = DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            csv_record = {
                **record,
                "context": json.dumps(record["context"], ensure_ascii=True, sort_keys=True),
                "metadata": json.dumps(record["metadata"], ensure_ascii=True, sort_keys=True),
            }
            writer.writerow(csv_record)
        return stream.getvalue()

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> TraceBuffer:
        """Build a trace buffer from serialized step records."""
        instance = cls()
        for record in records:
            instance.append(
                SimulationStepResult(
                    step_index=int(record["step_index"]),
                    context=record["context"],
                    chosen_arm=record["chosen_arm"],
                    reward=float(record["reward"]),
                    cumulative_reward=float(record["cumulative_reward"]),
                    cumulative_regret=float(record["cumulative_regret"]),
                    metadata=dict(record.get("metadata", {})),
                )
            )
        return instance

    @classmethod
    def from_json(cls, payload: str) -> TraceBuffer:
        """Deserialize trace buffer from JSON payload."""
        return cls.from_records(list(json.loads(payload)))


def filter_trace_records(records: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Filter trace records by case-insensitive token search."""
    cleaned = query.strip().lower()
    if not cleaned:
        return records
    result: list[dict[str, Any]] = []
    for record in records:
        haystack = json.dumps(record, ensure_ascii=True, sort_keys=True).lower()
        if cleaned in haystack:
            result.append(record)
    return result


def _to_json_value(value: Any) -> Any:
    """Convert unknown objects into a JSON-safe representation."""
    if isinstance(value, dict):
        return {str(key): _to_json_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_json_value(item) for item in value]

    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)
