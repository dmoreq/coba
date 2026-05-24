"""Bernoulli Thompson Sampling policy."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from web.contracts import BanditPolicy, DebugSnapshotProvider


class ThompsonSamplingPolicy(BanditPolicy[Any, Any], DebugSnapshotProvider):
    """Beta-Bernoulli Thompson Sampling for rewards in [0, 1]."""

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0, seed: int = 0) -> None:
        if prior_alpha <= 0.0 or prior_beta <= 0.0:
            raise ValueError("prior_alpha and prior_beta must be > 0")
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self._seed = seed
        self._rng = random.Random(seed)
        self._successes: dict[Any, float] = {}
        self._failures: dict[Any, float] = {}

    def reset(self) -> None:
        self._rng = random.Random(self._seed)
        self._successes.clear()
        self._failures.clear()

    def select_arm(self, context: Any, arms: Sequence[Any]) -> Any:
        _ = context
        if not arms:
            raise ValueError("ThompsonSamplingPolicy requires at least one arm")
        arm_list = list(arms)
        self._ensure_arms(arm_list)
        return max(
            arm_list,
            key=lambda arm: self._rng.betavariate(
                self.prior_alpha + self._successes[arm],
                self.prior_beta + self._failures[arm],
            ),
        )

    def update(self, context: Any, arm: Any, reward: float) -> None:
        _ = context
        self._ensure_arms([arm])
        clipped = min(1.0, max(0.0, reward))
        self._successes[arm] += clipped
        self._failures[arm] += 1.0 - clipped

    def _ensure_arms(self, arms: Sequence[Any]) -> None:
        for arm in arms:
            if arm not in self._successes:
                self._successes[arm] = 0.0
                self._failures[arm] = 0.0

    def get_debug_snapshot(self) -> dict[str, Any]:
        arms = {}
        for arm in self._successes:
            alpha = self.prior_alpha + self._successes[arm]
            beta = self.prior_beta + self._failures[arm]
            arms[str(arm)] = {
                "successes": self._successes[arm],
                "failures": self._failures[arm],
                "alpha_posterior": alpha,
                "beta_posterior": beta,
                "mean": alpha / (alpha + beta),
            }
        return {
            "policy": "thompson",
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
            "arms": arms,
        }
