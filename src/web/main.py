"""Thin Flet entry point for COBA — delegates to AppShell."""

from __future__ import annotations

from typing import Any

from web.app import AppShell
from web.curriculum import (
    LessonProgressState,
    get_lesson_by_policy,
)
from web.policy_factory import build_policy
from web.simulator import DiscreteSimulator
from web.state import RunConfig
from web.ui.preferences import PreferencesStore, UserPreferences
from web.ui.run_controls import RunController
from web.worlds import create_world, get_world_config

try:
    import flet as ft
except ModuleNotFoundError:  # pragma: no cover
    ft = None  # type: ignore[assignment]


class _SimSession:
    """Mutable session state that lives across route changes."""

    def __init__(self, prefs: UserPreferences) -> None:
        self.prefs = prefs
        self._reset_simulator()

    def _reset_simulator(self) -> None:
        world = create_world(self.prefs.world_id)
        config = get_world_config(self.prefs.world_id)
        feature_order = tuple(f.name for f in config.features)
        policy = build_policy(
            self.prefs.policy_id,
            feature_order=feature_order,
            seed=0,
        )
        self.simulator = DiscreteSimulator(
            policy=policy,
            world=world,
            config=RunConfig(seed=0, horizon=10000),
        )
        self.simulator.reset()
        self.controller = RunController()
        self._cancel_autoplay = False

        lesson = None
        try:
            lesson = get_lesson_by_policy(self.prefs.policy_id)
        except KeyError:
            pass
        self.lesson_config = lesson
        self.lesson_progress = (
            LessonProgressState(lesson_id=lesson.lesson_id, current_stage=1, completed=False)
            if lesson
            else None
        )

    def sync_prefs(self, prefs: UserPreferences) -> bool:
        changed = prefs.world_id != self.prefs.world_id or prefs.policy_id != self.prefs.policy_id
        self.prefs = prefs
        if changed:
            self._reset_simulator()
        return changed

    def do_step(self) -> None:
        if self.controller.state.mode == "running":
            self.controller.step()
        elif self.controller.state.mode in ("idle", "paused"):
            self.controller.play()
            self.controller.step()
            self.controller.pause()

    def do_play(self) -> None:
        self.controller.play()
        self._cancel_autoplay = False

    def do_pause(self) -> None:
        self.controller.pause()
        self._cancel_autoplay = True

    def do_reset(self) -> None:
        self._cancel_autoplay = True
        self.simulator.reset()
        self.controller.reset()
        if self.lesson_config:
            self.lesson_progress = LessonProgressState(
                lesson_id=self.lesson_config.lesson_id,
                current_stage=1,
                completed=False,
            )


# ── Global state refs (set in main()) ──────────────────────────────
_page: Any = None
_session: _SimSession | None = None
_pref_store: PreferencesStore | None = None
_shell: AppShell | None = None


def main(page: Any) -> None:
    """Application entry point."""
    global _page, _session, _pref_store, _shell
    if ft is None:
        raise RuntimeError("Flet is not installed.")

    _page = page
    _pref_store = PreferencesStore()
    prefs = _pref_store.load()
    _session = _SimSession(prefs)

    _shell = AppShell(page, _session, _pref_store)
    _shell.apply_theme("light")

    page.title = "COBA — Contextual Bandit Lab"

    def on_disconnect(event: Any) -> None:
        if _pref_store and _session:
            _pref_store.save(_session.prefs)

    page.on_disconnect = on_disconnect
    _shell._refresh_view()
    page.update()


def run() -> None:
    if ft is None:
        raise RuntimeError("Flet is not installed. Install it with `pip install flet`.")
    ft.run(main, view=ft.AppView.WEB_BROWSER, port=8550)


if __name__ == "__main__":
    run()
