"""UCB1 policy for discrete arms."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

from web.contracts import BanditPolicy, DebugSnapshotProvider


class UCB1Policy(BanditPolicy[Any, Any], DebugSnapshotProvider):
    """Upper Confidence Bound policy with one-time warm start per arm."""

    def __init__(self, alpha: float = 1.0, seed: int = 0) -> None:
        if alpha <= 0.0:
            raise ValueError("alpha must be > 0")
        self.alpha = alpha
        self._seed = seed
        self._rng = random.Random(seed)
        self._reward_sum: dict[Any, float] = {}
        self._pulls: dict[Any, int] = {}
        self._total_pulls = 0

    def reset(self) -> None:
        self._rng = random.Random(self._seed)
        self._reward_sum.clear()
        self._pulls.clear()
        self._total_pulls = 0

    def select_arm(self, context: Any, arms: Sequence[Any]) -> Any:
        _ = context
        if not arms:
            raise ValueError("UCB1Policy requires at least one arm")
        arm_list = list(arms)
        self._ensure_arms(arm_list)

        # Pull each arm at least once before using confidence terms.
        cold_arms = [arm for arm in arm_list if self._pulls[arm] == 0]
        if cold_arms:
            return self._rng.choice(cold_arms)

        log_total = math.log(max(1, self._total_pulls))
        return max(
            arm_list,
            key=lambda arm: (
                self._mean_reward(arm)
                + self.alpha * math.sqrt((2.0 * log_total) / float(self._pulls[arm]))
            ),
        )

    def update(self, context: Any, arm: Any, reward: float) -> None:
        _ = context
        self._ensure_arms([arm])
        self._pulls[arm] += 1
        self._reward_sum[arm] += reward
        self._total_pulls += 1

    def _ensure_arms(self, arms: Sequence[Any]) -> None:
        for arm in arms:
            if arm not in self._pulls:
                self._pulls[arm] = 0
                self._reward_sum[arm] = 0.0

    def _mean_reward(self, arm: Any) -> float:
        pulls = self._pulls[arm]
        if pulls == 0:
            return 0.0
        return self._reward_sum[arm] / float(pulls)

    def get_debug_snapshot(self) -> dict[str, Any]:
        arms = {}
        log_total = math.log(max(1, self._total_pulls))
        for arm in self._pulls:
            bonus = math.sqrt((2.0 * log_total) / float(max(1, self._pulls[arm])))
            arms[str(arm)] = {
                "pulls": self._pulls[arm],
                "mean_reward": self._mean_reward(arm),
                "ucb_bonus": self.alpha * bonus,
                "score": self._mean_reward(arm) + self.alpha * bonus,
            }
        return {
            "policy": "ucb1",
            "alpha": self.alpha,
            "total_pulls": self._total_pulls,
            "arms": arms,
        }
