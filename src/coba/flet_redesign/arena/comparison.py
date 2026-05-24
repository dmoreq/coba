"""In-memory run snapshot comparison helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class RunSnapshot:
    """One completed arena run snapshot."""

    run_id: str
    policy_id: str
    world_id: str
    cumulative_reward: float
    cumulative_regret: float
    replay_payload: dict[str, Any]
    created_at: str


class ArenaRunStore:
    """Hold current and previous run snapshots for quick comparison."""

    def __init__(self) -> None:
        self.current: RunSnapshot | None = None
        self.previous: RunSnapshot | None = None

    def commit(self, snapshot: RunSnapshot) -> None:
        self.previous = self.current
        self.current = snapshot

    def build_snapshot(
        self,
        run_id: str,
        policy_id: str,
        world_id: str,
        cumulative_reward: float,
        cumulative_regret: float,
        replay_payload: dict[str, Any],
    ) -> RunSnapshot:
        return RunSnapshot(
            run_id=run_id,
            policy_id=policy_id,
            world_id=world_id,
            cumulative_reward=cumulative_reward,
            cumulative_regret=cumulative_regret,
            replay_payload=replay_payload,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
