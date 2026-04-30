"""
RewardNormalizer — running normalization for reward signals.

Bandit arm models (especially Thompson Sampling and Logistic) assume rewards
in [0, 1]. Real business metrics (revenue, dwell time, click depth) are rarely
in that range. RewardNormalizer tracks running statistics and scales each reward
before it reaches the bandit, keeping the math sound without requiring the caller
to know about normalization internals.

Two modes:
  - "minmax": scales to [0, 1] using running min and max.
  - "zscore": standardizes to mean=0, std=1 (better for linear contextual bandits
              where reward range doesn't need to be bounded).

Both modes use an exponential moving update so older observations are down-weighted
and the normalizer adapts to distribution shifts over time.
"""

from __future__ import annotations

import math
from typing import Literal

from loguru import logger


class RewardNormalizer:
    """Tracks running reward statistics and normalizes incoming rewards.

    Designed to sit between the domain layer and ClusterBandit.update() so
    business-scale rewards (revenue, engagement time, etc.) are automatically
    scaled before the bandit's math sees them.

    Args:
        mode: "minmax" scales to [0, 1]; "zscore" standardizes to mean≈0, std≈1.
        decay: Exponential decay factor for running statistics (0 < decay < 1).
               Higher → slower adaptation (more history). Default 0.999.
        clip: If True, clamp minmax output to [0, 1] even when a new sample
              exceeds the observed min/max. Default True.

    Example::

        normalizer = RewardNormalizer(mode="minmax")
        for reward in revenue_stream:
            normed = normalizer.update_and_normalize(reward)
            bandit.update(context=ctx, arm=arm, reward=normed)
    """

    def __init__(
        self,
        mode: Literal["minmax", "zscore"] = "minmax",
        decay: float = 0.999,
        clip: bool = True,
    ) -> None:
        if mode not in ("minmax", "zscore"):
            raise ValueError(f"mode must be 'minmax' or 'zscore', got {mode!r}")
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}")

        self.mode = mode
        self.decay = decay
        self.clip = clip

        # minmax state — initialised on first observation
        self._min: float = 0.0
        self._max: float = 0.0
        self._minmax_init: bool = False

        # zscore state (exponential moving mean + variance)
        self._mean: float = 0.0
        self._var: float = 1.0
        self._n: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_and_normalize(self, reward: float) -> float:
        """Update running statistics with this reward and return its normalized value.

        Args:
            reward: Raw reward value (any finite float).

        Returns:
            Normalized reward. For "minmax" in [0, 1] (clipped if self.clip=True).
            For "zscore" in approximately (-3, 3) depending on distribution.
        """
        if not math.isfinite(reward):
            raise ValueError(f"reward must be finite, got {reward}")
        self._update_stats(reward)
        return self._normalize(reward)

    def normalize(self, reward: float) -> float:
        """Normalize using current statistics without updating them.

        Useful for normalizing evaluation data without contaminating the
        running statistics used for training.

        Args:
            reward: Raw reward value.

        Returns:
            Normalized reward using the last observed statistics.
        """
        if not math.isfinite(reward):
            raise ValueError(f"reward must be finite, got {reward}")
        return self._normalize(reward)

    @property
    def is_fitted(self) -> bool:
        """True after at least one observation has been processed."""
        return self._n > 0

    def reset(self) -> None:
        """Clear all running statistics."""
        self._min = 0.0
        self._max = 0.0
        self._minmax_init = False
        self._mean = 0.0
        self._var = 1.0
        self._n = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_stats(self, reward: float) -> None:
        self._n += 1
        alpha = 1.0 - self.decay  # weight for the new observation

        if self.mode == "minmax":
            if not self._minmax_init:
                self._min = self._max = reward
                self._minmax_init = True
            else:
                # Exponential moving min/max — decay toward the new value
                self._min = self.decay * self._min + alpha * min(reward, self._min)
                self._max = self.decay * self._max + alpha * max(reward, self._max)
                # Hard update if new extremes are seen
                if reward < self._min:
                    self._min = reward
                if reward > self._max:
                    self._max = reward
        else:
            # Exponential moving mean and variance
            delta = reward - self._mean
            self._mean += alpha * delta
            self._var = self.decay * self._var + alpha * delta * (reward - self._mean)
            self._var = max(self._var, 0.0)  # prevent fp drift below zero

    def _normalize(self, reward: float) -> float:
        if self._n == 0:
            logger.warning(
                "RewardNormalizer.normalize() called before any observations — returning reward as-is"
            )
            return reward

        if self.mode == "minmax":
            span = self._max - self._min
            if span < 1e-12:
                return 0.5
            normed = (reward - self._min) / span
            return float(max(0.0, min(1.0, normed))) if self.clip else float(normed)
        else:
            std = math.sqrt(self._var)
            if std < 1e-8:
                # Near-constant stream — all values are at the mean, return 0
                return 0.0
            return float((reward - self._mean) / std)
