"""Per-page session state accessors.

Stored on page.data with string keys to ensure multi-user web isolation.
app.py and main.py both need these — this module breaks the circular import.
"""

from __future__ import annotations

from typing import Any

from web.ui.preferences import PreferencesStore

_SESSION_KEY = "__coba_session"
_PREFS_KEY = "__coba_prefs"
_SHELL_KEY = "__coba_shell"


def get_session(page: Any) -> Any:
    """Get the current user's _SimSession from page.data."""
    return page.data.get(_SESSION_KEY)


def set_session(page: Any, session: Any) -> None:
    page.data[_SESSION_KEY] = session


def get_prefs(page: Any) -> PreferencesStore | None:
    """Get the current user's PreferencesStore from page.data."""
    return page.data.get(_PREFS_KEY)


def set_prefs(page: Any, prefs: PreferencesStore) -> None:
    page.data[_PREFS_KEY] = prefs


def get_shell(page: Any) -> Any:
    """Get the current user's AppShell from page.data."""
    return page.data.get(_SHELL_KEY)


def set_shell(page: Any, shell: Any) -> None:
    page.data[_SHELL_KEY] = shell
