"""
Sliding-Window LinUCB — LinUCB with a fixed-size FIFO observation window.

Motivation:
  Exponential discount (gamma < 1) handles *gradual* drift well but struggles
  when reward distributions change abruptly (step-function shifts). A
  sliding-window approach keeps exactly the ``window_size`` most recent
  observations, discarding older data entirely. This gives a clean separation:
  only the last W observations inform the current estimate.

  Compare with LinUCBArmModel (full history with optional gamma discount).

  - Use SlidingWindowLinUCBArmModel when drift is abrupt (regime changes).
  - Use LinUCBArmModel with gamma < 1 when drift is gradual.

Design:
  A FIFO deque buffers the ``window_size`` most recent (x, reward, weight)
  tuples. After each update, if the buffer is full a full refit is triggered
  (reset ridge → replay buffer). When the buffer is not full, incremental
  Sherman-Morrison updates are used instead.

  The full-refit path is O(W * d²) which is more expensive than incremental
  updates but is only triggered when the window rotates (every observation
  once the window is full).

Args:
    arm: Arm identifier.
    n_features: Context vector dimensionality.
    window_size: Number of most-recent observations to keep. Default 200.
    alpha: Exploration parameter (UCB width multiplier).
    l2_lambda: L2 regularization strength.
    rng: NumPy random generator.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from coba.policies.base import BaseArmModel
from coba.policies.ridge import RidgeRegression
from coba.types import Arm


class SlidingWindowLinUCBArmModel(BaseArmModel):
    """Per-arm LinUCB with a fixed-size observation window.

    Args:
        arm: Arm identifier.
        n_features: Context vector dimensionality.
        window_size: Maximum number of observations retained. Default 200.
        alpha: Exploration parameter (UCB width multiplier).
        l2_lambda: L2 regularization strength.
        gamma: Exponential discount factor within the window (1.0 = no discount).
        rng: NumPy random generator.
    """

    def __init__(
        self,
        arm: Arm,
        n_features: int,
        window_size: int = 200,
        alpha: float = 1.0,
        l2_lambda: float = 1.0,
        gamma: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(arm, rng or np.random.default_rng())
        self.n_features = n_features
        self.window_size = window_size
        self.alpha = alpha
        self.l2_lambda = l2_lambda
        self.gamma = gamma

        self._ridge = RidgeRegression(n_features=n_features, l2_lambda=l2_lambda, gamma=gamma)
        self._buffer: deque[tuple[np.ndarray, float, float]] = deque(maxlen=window_size)

    def score(self, x: np.ndarray) -> float:
        """Compute LinUCB score over the sliding window.

        Args:
            x: Context vector, shape (n_features,).
        Returns:
            UCB score.
        """
        expected_reward, ucb_width = self.score_decomposed(x)
        return expected_reward + ucb_width

    def score_decomposed(self, x: np.ndarray) -> tuple[float, float]:
        """Return (expected_reward, ucb_width) separately.

        Args:
            x: Context vector, shape (n_features,).
        Returns:
            Tuple of (expected_reward, ucb_width).
        """
        a_inv_x = self._ridge.A_inv @ x
        expected_reward = float(x @ self._ridge.beta)
        ucb_width = self.alpha * float(np.sqrt(max(x @ a_inv_x, 0.0)))
        return expected_reward, ucb_width

    def update(self, x: np.ndarray, reward: float, weight: float = 1.0) -> None:
        """Add observation; if the buffer was full, evict the oldest and refit.

        Args:
            x: Context vector, shape (n_features,).
            reward: Observed reward.
            weight: IPS importance weight.
        """
        was_full = len(self._buffer) == self.window_size
        self._buffer.append((x.copy(), reward, weight))

        if was_full:
            # Oldest observation was evicted — full refit from current buffer
            self._refit_from_buffer()
        else:
            # Buffer not yet full — incremental SM update is valid
            self._ridge.update(x, reward, weight)

        self.is_fitted = True

    def update_batch(
        self, x_batch: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
    ) -> None:
        """Batch update by iterating single-step updates."""
        ws = np.ones(len(y), dtype=np.float64) if weights is None else weights
        for xi, yi, wi in zip(x_batch, y, ws):
            self.update(xi, float(yi), float(wi))

    def reset(self) -> None:
        """Clear buffer and ridge model."""
        self._buffer.clear()
        self._ridge.reset()
        self.is_fitted = False

    @property
    def n_obs(self) -> int:
        return len(self._buffer)

    @property
    def beta(self) -> np.ndarray:
        return self._ridge.beta

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _refit_from_buffer(self) -> None:
        """Refit ridge from scratch using the current buffer contents."""
        self._ridge.reset()
        xs = [x for x, _, _ in self._buffer]
        ys = np.array([r for _, r, _ in self._buffer])
        ws = np.array([w for _, _, w in self._buffer])
        self._ridge.update_batch(np.stack(xs), ys, ws)
