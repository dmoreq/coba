"""Lightweight pub/sub event bus for cross-component communication.

Components subscribe to events and are notified when state changes occur.
"""

from __future__ import annotations

import asyncio
from typing import Any
from collections.abc import Callable


class EventBus:
    """Simple pub/sub event bus with async dispatch."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[..., None]]] = {}

    def subscribe(self, event_name: str, callback: Callable[..., None]) -> None:
        """Subscribe a callback to an event."""
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        self._subscribers[event_name].append(callback)

    def unsubscribe(self, event_name: str, callback: Callable[..., None]) -> None:
        """Unsubscribe a callback from an event."""
        if event_name in self._subscribers:
            self._subscribers[event_name] = [
                cb for cb in self._subscribers[event_name] if cb is not callback
            ]

    def emit(self, event_name: str, **data: Any) -> None:
        """Emit an event, calling all subscribers synchronously."""
        callbacks = self._subscribers.get(event_name, [])
        for callback in callbacks:
            try:
                callback(**data)
            except Exception:
                import traceback

                traceback.print_exc()

    async def emit_async(self, event_name: str, **data: Any) -> None:
        """Emit an event asynchronously, running all callbacks concurrently."""
        callbacks = self._subscribers.get(event_name, [])
        tasks = []
        for callback in callbacks:
            if asyncio.iscoroutinefunction(callback):
                tasks.append(callback(**data))
            else:
                callback(**data)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def clear(self) -> None:
        """Remove all subscribers."""
        self._subscribers.clear()


class Events:
    """Standard event names used across the application."""

    STEP_COMPLETED = "step_completed"
    ARM_SELECTED = "arm_selected"
    REWARD_RECEIVED = "reward_received"
    KNOWLEDGE_UPDATED = "knowledge_updated"
    THEME_CHANGED = "theme_changed"
    WORLD_CHANGED = "world_changed"
    POLICY_CHANGED = "policy_changed"
    RESET_TRIGGERED = "reset_triggered"
