"""Base layout protocol for the COBA web app."""

from __future__ import annotations

from typing import Any, Protocol


class LayoutProtocol(Protocol):
    """Protocol for layout builders."""

    def build(self, page: Any, view_model: Any, session: Any) -> Any:
        """Build the layout from a view model and session state."""
        ...
