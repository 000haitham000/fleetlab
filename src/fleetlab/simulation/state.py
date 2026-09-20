"""Execution state: the mutable half of the framework.

Everything else in ``fleetlab`` is immutable, and this module is the deliberate
exception. That is not an inconsistency; it is the point.

Two activities in a simulation-based study want opposite things:

============  ===========  ==================================================
activity      wants        because
============  ===========  ==================================================
simulation    mutable      one timeline, monotone progress; the clock really
                           does advance and a vehicle really does arrive
search        immutable    millions of throwaway candidates, every one of
                           which must be discardable for free
============  ===========  ==================================================

The earlier Java design did both jobs with one ``Vehicle`` object, which is why
it felt natural in one mode and awkward in the other -- and why ``currentTime``,
``currentStop`` and per-action statuses sat so uneasily beside ``addRequest``.

Here they are separate. :class:`ExecutionState` owns the clock and reality and
is freely mutable. The optimiser works on immutable
:class:`~fleetlab.domain.solution.Solution` objects and never sees this class;
it receives a frozen snapshot through
:class:`~fleetlab.timing.context.VehicleStart`. The simulator stops pretending
to be a solution representation, and the solution representation stops
pretending to be a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fleetlab.domain.solution import Solution
from fleetlab.timing.context import VehicleStart
from fleetlab.timing.evaluator import evaluate_route

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fleetlab.domain.ids import RequestId, StopId, VehicleId
    from fleetlab.domain.units import Instant
    from fleetlab.study import Study
    from fleetlab.timing.timing import StopTiming


@dataclass
class VehicleExecution:
    """What one vehicle has actually done. Mutable by design.

    Attributes:
        vehicle: Which vehicle.
        position: Where it is, or where it last was.
        free_at: When it becomes available for the next action.
        departed_at: When it left its depot, or ``None`` if it has not.
        completed: Recorded timings of the actions it has finished, in order.
        served: Requests whose dropoff it has completed.
        onboard: Requests it has picked up but not yet delivered.
    """

    vehicle: VehicleId
    position: StopId
    free_at: Instant
    departed_at: Instant | None = None
    completed: list[StopTiming] = field(default_factory=list)
    served: set[RequestId] = field(default_factory=set)
    onboard: set[RequestId] = field(default_factory=set)

    @property
    def actions_done(self) -> int:
        """How many actions this vehicle has completed."""
        return len(self.completed)

    def snapshot(self, *, allow_just_in_time: bool) -> VehicleStart:
        """Freeze into the immutable form the evaluator consumes."""
        return VehicleStart(
            stop=self.position,
            ready_at=self.free_at,
            allow_just_in_time=allow_just_in_time and self.departed_at is None,
            committed_timings=tuple(self.completed),
            departed_at=self.departed_at,
        )


class ExecutionState:
    """The clock, the fleet's real positions, and the plan currently in force.

    Mutated as the simulation advances. Never handed to an optimiser: call
    :meth:`freeze` for that.
    """

    __slots__ = ("_now", "_plan", "_study", "_vehicles")

    def __init__(self, study: Study, *, start_at: Instant = 0.0) -> None:
        """Begin a simulation with every vehicle at its depot and no plan."""
        self._study = study
        self._now = start_at
        self._plan = study.empty_solution(())
        self._vehicles: dict[VehicleId, VehicleExecution] = {
            vehicle_id: VehicleExecution(
                vehicle=vehicle_id,
                position=vehicle.start_stop,
                free_at=max(start_at, vehicle.available_from),
            )
            for vehicle_id, vehicle in study.problem.vehicles.items()
        }

    # ------------------------------------------------------------- inspection

    @property
    def now(self) -> Instant:
        """The current simulated instant."""
        return self._now

    @property
    def plan(self) -> Solution:
        """The plan currently in force."""
        return self._plan

    def vehicle(self, vehicle_id: VehicleId) -> VehicleExecution:
        """One vehicle's execution record."""
        return self._vehicles[vehicle_id]

    def served_requests(self) -> frozenset[RequestId]:
        """Requests fully delivered so far."""
        served: set[RequestId] = set()
        for execution in self._vehicles.values():
            served |= execution.served
        return frozenset(served)

    def onboard_requests(self) -> frozenset[RequestId]:
        """Requests picked up but not yet delivered."""
        aboard: set[RequestId] = set()
        for execution in self._vehicles.values():
            aboard |= execution.onboard
        return frozenset(aboard)

    # ------------------------------------------------------------------ clock

    def advance_to(self, instant: Instant) -> int:
        """Run the plan forward to ``instant``, completing whatever falls due.

        Args:
            instant: The time to advance to. Going backwards is a programmer
                error.

        Returns:
            How many actions were completed during the advance.

        Raises:
            ValueError: If ``instant`` is before the current time.
        """
        if instant < self._now:
            msg = f"Cannot advance the clock backwards: {self._now} -> {instant}"
            raise ValueError(msg)

        completed = 0
        for schedule in self._plan.routes:
            execution = self._vehicles[schedule.vehicle]
            timing = evaluate_route(schedule, self._study.context)
            if timing.stops and execution.departed_at is None:
                execution.departed_at = timing.depot_departure

            for position in range(execution.actions_done, len(schedule.actions)):
                stop_timing = timing.stops[position]
                if stop_timing.departure > instant:
                    break
                action = schedule.actions[position]
                execution.completed.append(stop_timing)
                execution.position = action.stop
                execution.free_at = stop_timing.departure
                if action.is_pickup:
                    execution.onboard.add(action.request)
                else:
                    execution.onboard.discard(action.request)
                    execution.served.add(action.request)
                completed += 1

        self._now = instant
        for execution in self._vehicles.values():
            execution.free_at = max(execution.free_at, self._now)
        return completed

    # ------------------------------------------------------------- snapshots

    def freeze(self, *, allow_just_in_time: bool = True) -> Mapping[VehicleId, VehicleStart]:
        """The immutable start states an optimiser needs.

        This is the boundary between the mutable simulator and the pure search:
        everything the optimiser learns about reality arrives through this
        mapping, and nothing it does can reach back.
        """
        return {
            vehicle_id: execution.snapshot(allow_just_in_time=allow_just_in_time)
            for vehicle_id, execution in self._vehicles.items()
        }

    def rebased_study(self) -> Study:
        """A study whose vehicles start from where they actually are now."""
        return self._study.rebased(self.freeze())

    # ------------------------------------------------------------- committing

    def commit(self, plan: Solution) -> None:
        """Adopt a new plan.

        Raises:
            ValueError: If the plan rewrites history -- that is, if it changes an
                action a vehicle has already performed. Catching this here is
                what stops an optimiser bug from silently producing a
                simulation that could not have happened.
        """
        for schedule in plan.routes:
            execution = self._vehicles[schedule.vehicle]
            done = execution.actions_done
            if len(schedule.actions) < done:
                msg = (
                    f"Plan for {schedule.vehicle!r} drops {done - len(schedule.actions)} "
                    "action(s) the vehicle has already performed."
                )
                raise ValueError(msg)
            current = self._plan.route_for(schedule.vehicle) if self._plan.routes else None
            if current is not None and current.actions[:done] != schedule.actions[:done]:
                msg = (
                    f"Plan for {schedule.vehicle!r} reorders actions the vehicle has "
                    "already performed."
                )
                raise ValueError(msg)
        self._plan = plan

    def initial_plan(self, plan: Solution) -> None:
        """Adopt the first plan, before anything has been executed."""
        self._plan = plan

    def __str__(self) -> str:
        done = sum(execution.actions_done for execution in self._vehicles.values())
        return (
            f"ExecutionState(t={self._now:.1f}, {done} actions done, "
            f"{len(self.served_requests())} served, {len(self.onboard_requests())} aboard)"
        )
