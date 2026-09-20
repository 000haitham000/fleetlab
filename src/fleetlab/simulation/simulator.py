"""The simulation loop.

The shape a simulation-based optimisation study actually has::

    while there is another decision epoch:
        advance reality to it              # mutable
        freeze what the operator can see   # boundary
        re-plan                            # pure
        commit the new plan                # mutable

Everything the optimiser learns arrives through the freeze, and nothing it does
reaches back. So an algorithm written for the static case runs unchanged in the
dynamic one: it is handed a study, it returns a solution, and it never learns
that a clock exists.

Decision epochs come from two sources. Requests become known over time, and each
release is an epoch because it is new information. A fixed cadence adds the rest,
because conditions drift even when nothing new arrives. A study that wants
event-driven epochs only can set the cadence to ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fleetlab.simulation.locking import HorizonLocking
from fleetlab.simulation.state import ExecutionState

if TYPE_CHECKING:
    from fleetlab.domain.ids import RequestId
    from fleetlab.domain.schedule import Schedule
    from fleetlab.domain.solution import Solution
    from fleetlab.domain.units import Duration, Instant
    from fleetlab.search.base import Algorithm
    from fleetlab.simulation.locking import LockingPolicy
    from fleetlab.study import Study


@dataclass(frozen=True, slots=True)
class EpochRecord:
    """What happened at one decision epoch.

    Attributes:
        instant: When it occurred.
        newly_released: Requests that became known at or before it and were not
            known at the previous epoch.
        actions_completed: Actions executed since the previous epoch.
        planned: Requests assigned in the plan adopted here.
        unassigned: Requests known but unassigned after re-planning.
        cost: Cost of the adopted plan.
        replan_seconds: Wall-clock time the optimiser took.
    """

    instant: Instant
    newly_released: tuple[RequestId, ...]
    actions_completed: int
    planned: int
    unassigned: int
    cost: float
    replan_seconds: float


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """The outcome of a simulated day.

    Attributes:
        final_plan: The plan in force when the horizon ended.
        epochs: One record per decision epoch.
        served: Requests fully delivered.
        never_served: Requests that were released but never delivered.
        total_replan_seconds: Wall-clock time spent optimising.
    """

    final_plan: Solution
    epochs: tuple[EpochRecord, ...]
    served: frozenset[RequestId]
    never_served: frozenset[RequestId]
    total_replan_seconds: float = 0.0
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def service_rate(self) -> float:
        """Share of released requests that were delivered."""
        total = len(self.served) + len(self.never_served)
        return len(self.served) / total if total else 1.0

    def describe(self) -> str:
        """A multi-line report for a run log."""
        return "\n".join(
            [
                f"{len(self.epochs)} decision epochs, "
                f"{self.total_replan_seconds:.2f}s spent re-planning",
                f"{len(self.served)} served, {len(self.never_served)} never served "
                f"({self.service_rate:.1%})",
                f"final plan: {self.final_plan}",
            ]
        )

    def __str__(self) -> str:
        return f"SimulationResult({len(self.served)} served, {self.service_rate:.1%})"


@dataclass(frozen=True, slots=True)
class Simulator:
    """Runs a study forward through time, re-planning at decision epochs.

    Attributes:
        cadence: Minutes between periodic epochs, or ``None`` for event-driven
            epochs only.
        locking: How much of the plan is frozen against re-planning.
        allow_just_in_time: Whether vehicles still at their depot may delay
            departure to avoid arriving early.
        horizon_end: When to stop. Defaults to the instance's horizon.
    """

    cadence: Duration | None = 30.0
    locking: LockingPolicy = field(default_factory=HorizonLocking)
    allow_just_in_time: bool = True
    horizon_end: Instant | None = None

    def epochs(self, study: Study) -> tuple[Instant, ...]:
        """Every decision epoch, ascending.

        Release times supply the event-driven ones; the cadence fills in the
        rest.
        """
        end = self.horizon_end if self.horizon_end is not None else study.problem.horizon_end
        moments: set[Instant] = {0.0}
        moments.update(
            request.released_at
            for request in study.problem.requests.values()
            if request.released_at <= end
        )
        if self.cadence is not None and self.cadence > 0:
            step = self.cadence
            current = step
            while current < end:
                moments.add(current)
                current += step
        return tuple(sorted(moment for moment in moments if moment <= end))

    def run(self, study: Study, optimiser: Algorithm) -> SimulationResult:
        """Simulate a day, re-planning with ``optimiser`` at each epoch.

        Args:
            study: The instance, rules and objective.
            optimiser: Re-planner. It receives a study rebased on the fleet's
                real positions and a warm-start solution, and returns a plan.

        Returns:
            A :class:`SimulationResult`.
        """
        import time

        end = self.horizon_end if self.horizon_end is not None else study.problem.horizon_end
        state = ExecutionState(study)
        records: list[EpochRecord] = []
        known: set[RequestId] = set()
        total_replan = 0.0

        for epoch in self.epochs(study):
            completed = state.advance_to(epoch)

            released = study.released_by(epoch)
            newly = tuple(sorted(released - known))
            known |= set(released)

            rebased = study.rebased(state.freeze(allow_just_in_time=self.allow_just_in_time))
            warm = self._carry_forward(rebased, state, released)

            started = time.perf_counter()
            plan = optimiser.solve(rebased, warm).solution
            elapsed = time.perf_counter() - started
            total_replan += elapsed

            state.commit(plan)
            records.append(
                EpochRecord(
                    instant=epoch,
                    newly_released=newly,
                    actions_completed=completed,
                    planned=len(plan.assigned_requests()),
                    unassigned=len(plan.unassigned),
                    cost=rebased.cost(plan),
                    replan_seconds=elapsed,
                )
            )

        state.advance_to(end)
        served = state.served_requests()
        return SimulationResult(
            final_plan=state.plan,
            epochs=tuple(records),
            served=served,
            never_served=frozenset(known) - served,
            total_replan_seconds=total_replan,
            notes={"locking": self.locking.name, "cadence": str(self.cadence)},
        )

    def _carry_forward(
        self,
        study: Study,
        state: ExecutionState,
        released: frozenset[RequestId],
    ) -> Solution:
        """Build the warm start: keep what is locked, release the rest.

        This is where locking is applied. Everything up to the committed prefix
        stays exactly as it was; everything after it goes back into the pool for
        the optimiser to place afresh.
        """
        from fleetlab.domain.schedule import Schedule
        from fleetlab.domain.solution import Solution

        routes: list[Schedule] = []
        retained: set[RequestId] = set()

        for schedule in state.plan.routes:
            execution = state.vehicle(schedule.vehicle)
            executed = execution.actions_done
            if not schedule.actions:
                routes.append(Schedule(schedule.vehicle))
                continue

            timing = study.timing(schedule)
            frozen = self.locking.committed_prefix(
                schedule, timing, study.context, state.now, executed
            )
            frozen = _extend_to_whole_pairs(schedule, frozen)
            kept = schedule.actions[:frozen]
            routes.append(Schedule(schedule.vehicle, kept, frozen))
            retained.update(action.request for action in kept)

        pool = (released | state.onboard_requests()) - retained - state.served_requests()
        for vehicle_id in study.problem.vehicles:
            if not any(route.vehicle == vehicle_id for route in routes):
                routes.append(Schedule(vehicle_id))
        return Solution(tuple(routes), frozenset(pool))


def _extend_to_whole_pairs(schedule: Schedule, prefix: int) -> int:
    """Grow a prefix until no request is left half-present inside it.

    A frozen prefix ending between a pickup and its dropoff would strand the
    loadable: the pickup is locked in, but the dropoff has been released for
    re-planning onto another vehicle. Extending the prefix through the matching
    dropoff keeps every pair together, which is the invariant the whole
    framework rests on.
    """
    aboard = {action.request for action in schedule.actions[:prefix] if action.is_pickup} - {
        action.request for action in schedule.actions[:prefix] if action.is_dropoff
    }
    if not aboard:
        return prefix
    extended = prefix
    for position in range(prefix, len(schedule.actions)):
        action = schedule.actions[position]
        if action.is_dropoff and action.request in aboard:
            aboard.discard(action.request)
            extended = position + 1
            if not aboard:
                break
    return extended
