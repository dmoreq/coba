"""
Persistence utilities — save and load ClusterRouter state.

Uses joblib for efficient serialization of numpy arrays (the dominant
data type in the bandit's internal state). Falls back to Python pickle
for simple objects.

Usage:
  from coba.persistence import save_bandit, load_bandit

  save_bandit(bandit, "/models/cluster_bandit.joblib")
  bandit = load_bandit("/models/cluster_bandit.joblib")
"""

from pathlib import Path

import joblib
from loguru import logger

from coba.bandit import ClusterBandit


def save_bandit(bandit: ClusterBandit, path: str | Path) -> None:
    """Persist a ClusterBandit to disk using joblib compression.

    Args:
        bandit: The bandit instance to save.
        path: Output file path. Recommended extension: .joblib
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bandit, path, compress=3)
    logger.info("Bandit saved to {path}", path=str(path))


def load_bandit(path: str | Path) -> ClusterBandit:
    """Load a ClusterBandit from disk.

    Args:
        path: Path to the saved .joblib file.

    Returns:
        The deserialized ClusterBandit instance.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Bandit model file not found: {path}")
    bandit = joblib.load(path)
    logger.info("Bandit loaded from {path}", path=str(path))
    return bandit


# Backward-compatible aliases used by older examples.
save_model = save_bandit
load_model = load_bandit
