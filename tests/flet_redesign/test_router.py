"""Tests for Flet redesign route helpers."""

from __future__ import annotations

from web.router import AppRoute, get_route_spec, list_route_specs, normalize_route


def test_normalize_route_defaults_to_home() -> None:
    assert normalize_route(None) == AppRoute.HOME
    assert normalize_route("") == AppRoute.HOME
    assert normalize_route("  ") == AppRoute.HOME


def test_normalize_route_supports_top_level_pages() -> None:
    assert normalize_route("/lesson") == AppRoute.LESSON
    assert normalize_route("arena") == AppRoute.ARENA
    assert normalize_route("/sandbox/custom") == AppRoute.SANDBOX


def test_unknown_route_falls_back_to_home() -> None:
    assert normalize_route("/unknown/path") == AppRoute.HOME


def test_get_route_spec_maps_to_titles() -> None:
    assert get_route_spec("/").title == "Home"
    assert get_route_spec("/lesson/intro").title == "Lesson"


def test_list_route_specs_includes_all_routes() -> None:
    specs = list_route_specs()
    assert set(spec.route for spec in specs) == {
        AppRoute.HOME,
        AppRoute.LESSON,
        AppRoute.ARENA,
        AppRoute.SANDBOX,
        AppRoute.COMPARISON,
    }
