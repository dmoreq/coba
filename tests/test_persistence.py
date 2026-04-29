"""Tests for persistence utilities (save_bandit / load_bandit)."""

import pathlib

import numpy as np
import pytest

from coba import ClusterBandit
from coba.persistence import (
    load_bandit,
    load_model,
    save_bandit,
    save_model,
)

ARMS = [1.0, 1.1, 1.2, 1.5]


def _make_fitted_bandit() -> ClusterBandit:
    """Create a fully fitted ClusterBandit for persistence tests."""
    bandit = ClusterBandit(arms=ARMS, n_features=4, n_clusters=2, seed=0)
    rng = np.random.default_rng(0)
    bandit.fit_offline(
        contexts=rng.standard_normal((100, 4)),
        decisions=rng.choice(ARMS, 100),
        rewards=rng.uniform(0, 1, 100),
    )
    return bandit


class TestSaveBandit:
    def test_creates_file(self, tmp_path: pathlib.Path) -> None:
        """save_bandit() should create the file at the given path."""
        bandit = _make_fitted_bandit()
        out = tmp_path / "model.joblib"
        save_bandit(bandit, out)
        assert out.exists()

    def test_creates_parent_directories(self, tmp_path: pathlib.Path) -> None:
        """save_bandit() should create any missing parent directories."""
        bandit = _make_fitted_bandit()
        out = tmp_path / "nested" / "dir" / "model.joblib"
        save_bandit(bandit, out)
        assert out.exists()

    def test_accepts_string_path(self, tmp_path: pathlib.Path) -> None:
        """save_bandit() should accept a plain string path."""
        bandit = _make_fitted_bandit()
        out = str(tmp_path / "model.joblib")
        save_bandit(bandit, out)
        assert pathlib.Path(out).exists()


class TestLoadBandit:
    def test_raises_if_file_missing(self, tmp_path: pathlib.Path) -> None:
        """load_bandit() should raise FileNotFoundError for non-existent files."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_bandit(tmp_path / "does_not_exist.joblib")

    def test_returns_cluster_bandit_instance(self, tmp_path: pathlib.Path) -> None:
        """Loaded object should be an instance of ClusterBandit."""
        bandit = _make_fitted_bandit()
        path = tmp_path / "model.joblib"
        save_bandit(bandit, path)
        loaded = load_bandit(path)
        assert isinstance(loaded, ClusterBandit)


class TestRoundTrip:
    """Verify that save→load preserves the bandit's learned state."""

    def test_is_fitted_preserved(self, tmp_path: pathlib.Path) -> None:
        bandit = _make_fitted_bandit()
        path = tmp_path / "model.joblib"
        save_bandit(bandit, path)
        loaded = load_bandit(path)
        assert loaded.is_fitted

    def test_arms_preserved(self, tmp_path: pathlib.Path) -> None:
        bandit = _make_fitted_bandit()
        path = tmp_path / "model.joblib"
        save_bandit(bandit, path)
        loaded = load_bandit(path)
        assert loaded.arms == bandit.arms

    def test_decide_returns_valid_arm(self, tmp_path: pathlib.Path) -> None:
        """After round-trip, the bandit should still be able to make decisions."""
        bandit = _make_fitted_bandit()
        ctx = np.array([1.0, -0.5, 0.2, 0.8])
        original_decision = bandit.decide(ctx)

        path = tmp_path / "model.joblib"
        save_bandit(bandit, path)
        loaded = load_bandit(path)

        loaded_decision = loaded.decide(ctx)
        # Same model state → same arm selection
        assert loaded_decision.chosen_arm == original_decision.chosen_arm
        assert loaded_decision.chosen_arm in ARMS

    def test_stats_preserved(self, tmp_path: pathlib.Path) -> None:
        """Monitoring stats (n_clusters) should survive serialization."""
        bandit = _make_fitted_bandit()
        path = tmp_path / "model.joblib"
        save_bandit(bandit, path)
        loaded = load_bandit(path)
        assert loaded.n_clusters == bandit.n_clusters

    def test_compat_aliases_roundtrip(self, tmp_path: pathlib.Path) -> None:
        """Backward-compatible save_model/load_model aliases should work."""
        bandit = _make_fitted_bandit()
        path = tmp_path / "model_alias.joblib"
        save_model(bandit, path)
        loaded = load_model(path)
        assert loaded.is_fitted
        assert loaded.arms == bandit.arms
