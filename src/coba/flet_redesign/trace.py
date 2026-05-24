"""Trace utilities for simulation replay and export."""

from __future__ import annotations

import json
from dataclasses import asdict
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
