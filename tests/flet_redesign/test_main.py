"""Tests for Flet redesign app entrypoint behavior."""

from __future__ import annotations

import importlib

import pytest

flet_main_module = importlib.import_module("coba.flet_redesign.main")


def test_run_raises_when_flet_missing() -> None:
    if flet_main_module.ft is not None:
        pytest.skip("flet is installed in this environment")
    with pytest.raises(RuntimeError, match="Flet is not installed"):
        flet_main_module.run()
