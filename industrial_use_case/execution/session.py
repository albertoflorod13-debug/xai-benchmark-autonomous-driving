"""
session.py
===========
In-memory state for one demo run: the fixed start/target, the currently
active route, and the running history of steps, conflicts and replans.
Exposes a small set of methods that encapsulate the mutations that matter
for correctness -- in particular, telling a fresh replan apart from the
run's very first route (see record_replan()).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from ..planning.llm_rrt_planner import RouteResult
from .conflict_checker import ConflictResult

Point = Tuple[float, float]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ConflictLogEntry:
    """One conflict-triggering step, kept so the report generator can explain
    exactly this object without re-running detection."""
    step_index: int
    position: Point
    image_path: str
    conflict: ConflictResult
    timestamp: str


@dataclass
class RouteLogEntry:
    """One planned route within the run, wrapping a RouteResult with the
    reason plan_full_route() was called."""
    trigger: str  # "initial" | "conflict"
    route: RouteResult
    timestamp: str


@dataclass
class DemoSession:
    """Full state of one demo run, from the initial route to completion."""

    start: Point
    target: Point
    max_total_replans: int
    current_position: Point

    active_route: Optional[RouteResult] = None
    num_replans: int = 0
    step_counter: int = 0
    route_history: List[RouteLogEntry] = field(default_factory=list)
    conflict_log: List[ConflictLogEntry] = field(default_factory=list)
    finished: bool = False
    reached_target: bool = False

    def next_step_index(self) -> int:
        """Call once per physical step taken, before logging it."""
        self.step_counter += 1
        return self.step_counter

    def record_step(self, position: Point) -> None:
        """Commit the robot's new position after taking one physical step."""
        self.current_position = position

    def record_conflict(self, step_index: int, position: Point, image_path: str,
                         conflict: ConflictResult) -> None:
        self.conflict_log.append(ConflictLogEntry(
            step_index=step_index, position=position, image_path=image_path,
            conflict=conflict, timestamp=_utcnow_iso(),
        ))

    def record_replan(self, route: RouteResult, trigger: str) -> None:
        """
        Commit a freshly planned route into the session.

        `trigger` must be "initial" (the run's very first route) or
        "conflict" (every later replan) -- required with no default so a
        replan is never silently mislabeled. num_replans only counts real
        replans (trigger != "initial").
        """
        self.active_route = route
        if trigger != "initial":
            self.num_replans += 1
        self.route_history.append(RouteLogEntry(
            trigger=trigger, route=route, timestamp=_utcnow_iso(),
        ))

    def budget_exhausted(self) -> bool:
        return self.num_replans >= self.max_total_replans