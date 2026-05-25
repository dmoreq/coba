"""Edge case tests for new components — event bus, theme, charts, and view models."""

from __future__ import annotations

import pytest

from web.statemgmt.event_bus import EventBus, Events


def test_event_bus_emit_with_all_event_names():
    """All standard Events constants are valid event names."""
    bus = EventBus()
    collected: list[str] = []
    for attr_name in dir(Events):
        if attr_name.startswith("_"):
            continue
        event_name = getattr(Events, attr_name)
        if isinstance(event_name, str):
            bus.subscribe(event_name, lambda **__: collected.append(event_name))

    # Emit all event names
    for attr_name in dir(Events):
        if attr_name.startswith("_"):
            continue
        event_name = getattr(Events, attr_name)
        if isinstance(event_name, str):
            bus.emit(event_name)

    assert len(collected) > 0
    # All events should be standard event names
    for name in collected:
        assert isinstance(name, str)
        assert "_" in name or name.islower()


def test_event_bus_async_emit_with_no_coro():
    """emit_async works even when all callbacks are sync."""
    import asyncio

    bus = EventBus()
    results: list[int] = []

    bus.subscribe("evt", lambda **__: results.append(1))
    asyncio.run(bus.emit_async("evt"))
    assert results == [1]


def test_event_bus_subscribe_same_callback_twice():
    """Subscribing the same callback twice calls it twice per emit."""
    bus = EventBus()
    results: list[int] = []

    def handler(**_: object) -> None:
        results.append(1)

    bus.subscribe("evt", handler)
    bus.subscribe("evt", handler)
    bus.emit("evt")
    assert len(results) == 2


def test_event_bus_unsubscribe_nonexistent_callback():
    """Unsubscribing a callback that was never added does nothing."""
    bus = EventBus()

    def handler(**_: object) -> None:
        pass

    bus.unsubscribe("evt", handler)  # should not raise


def test_view_model_with_missing_world_id():
    """build_route_ui_model with invalid world_id raises KeyError."""
    from web.ui.preferences import UserPreferences

    prefs = UserPreferences(world_id="nonexistent_world", policy_id="random", speed="1x")
    with pytest.raises(KeyError):
        from web.ui.view_models import build_route_ui_model

        build_route_ui_model("/arena", prefs=prefs)


def test_preferences_defaults_are_reasonable():
    """Default preferences use real world and policy IDs."""
    from web.ui.preferences import UserPreferences

    prefs = UserPreferences()
    assert prefs.world_id == "rural_clinic"
    assert prefs.policy_id == "random"
    assert prefs.speed == "1x"


def test_param_controls_cats_policy():
    """CATS (continuous) policy exposes param controls."""
    from web.ui.param_controls import default_policy_param_controls

    controls = default_policy_param_controls("cats")
    control_keys = {c.key for c in controls}
    assert "action_min" in control_keys
    assert "action_max" in control_keys
