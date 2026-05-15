"""Tests for the public policy package exports."""

from __future__ import annotations

import coba.policies as policies


def test_policy_all_exports_are_defined() -> None:
    """Every name in __all__ should be importable from coba.policies."""
    missing = [name for name in policies.__all__ if not hasattr(policies, name)]
    assert missing == []
