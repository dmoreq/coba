"""Tests for Flet redesign shell stack composition."""

from __future__ import annotations

from coba.flet_redesign.shell import build_shell_stack


def test_home_route_yields_single_home_view() -> None:
    stack = build_shell_stack("/")
    assert len(stack) == 1
    assert stack[0].route == "/"
    assert stack[0].title == "Home"


def test_non_home_route_yields_home_plus_active() -> None:
    stack = build_shell_stack("/lesson/ucb1")
    assert len(stack) == 2
    assert stack[0].route == "/"
    assert stack[1].route == "/lesson"
    assert stack[1].title == "Lesson"


def test_unknown_route_falls_back_to_home_stack() -> None:
    stack = build_shell_stack("/nope")
    assert len(stack) == 1
    assert stack[0].route == "/"
