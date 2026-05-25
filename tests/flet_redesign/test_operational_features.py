"""Tests for drift timeline (checkpoint/preset modules removed as dead code)."""

from __future__ import annotations

from web.drift_monitor import DriftTimeline


def test_drift_timeline_records_events() -> None:
    timeline = DriftTimeline(delta=0.0, lambda_=0.2, alpha=0.8)
    values = [0.1, 0.1, 0.1, 0.9, 0.95, 1.0]
    detected = [
        timeline.update(step_index=index + 1, value=value) for index, value in enumerate(values)
    ]
    assert any(detected)
    assert timeline.indicators()["num_events"] >= 1
