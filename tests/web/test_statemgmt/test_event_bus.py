"""Tests for event bus."""

from __future__ import annotations


def test_subscribe_and_emit() -> None:
    from web.statemgmt.event_bus import EventBus

    bus = EventBus()
    results: list[str] = []

    def handler(**data: object) -> None:
        results.append(data.get("msg", ""))

    bus.subscribe("test_event", handler)
    bus.emit("test_event", msg="hello")
    assert results == ["hello"]


def test_multiple_subscribers() -> None:
    from web.statemgmt.event_bus import EventBus

    bus = EventBus()
    results: list[int] = []

    bus.subscribe("evt", lambda **_: results.append(1))
    bus.subscribe("evt", lambda **_: results.append(2))

    bus.emit("evt")
    assert sorted(results) == [1, 2]


def test_unsubscribe() -> None:
    from web.statemgmt.event_bus import EventBus

    bus = EventBus()
    results: list[int] = []

    def handler(**_: object) -> None:
        results.append(1)

    bus.subscribe("evt", handler)
    bus.emit("evt")
    assert len(results) == 1

    bus.unsubscribe("evt", handler)
    bus.emit("evt")
    assert len(results) == 1  # no change


def test_emit_with_no_subscribers_does_nothing() -> None:
    from web.statemgmt.event_bus import EventBus

    bus = EventBus()
    bus.emit("nonexistent")  # should not raise


def test_error_in_one_subscriber_does_not_block_others() -> None:
    from web.statemgmt.event_bus import EventBus

    bus = EventBus()
    results: list[int] = []

    def failing(**_: object) -> None:
        msg = 1 / 0  # noqa: F841

    def succeeding(**_: object) -> None:
        results.append(1)

    bus.subscribe("evt", failing)
    bus.subscribe("evt", succeeding)
    bus.emit("evt")
    assert results == [1]


def test_clear_removes_all_subscribers() -> None:
    from web.statemgmt.event_bus import EventBus

    bus = EventBus()
    results: list[int] = []

    bus.subscribe("evt", lambda **_: results.append(1))
    bus.clear()
    bus.emit("evt")
    assert results == []
