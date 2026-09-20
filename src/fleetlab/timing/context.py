"""Evaluation context: everything the evaluator needs besides the schedule itself.

Separating this from the schedule is what lets the *same* route be evaluated
under different assumptions -- a different service policy, a frozen traffic
profile, or the fleet's real position part-way through a simulated day -- without
touching the route.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fleetlab.timing.policy import DEFAULT_POLICY

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fleetlab.domain.ids import StopId, VehicleId
    from fleetlab.domain.problem import Problem
    from fleetlab.domain.units import Instant
    from fleetlab.timing.policy import ServicePolicy
    from fleetlab.timing.timing import StopTiming


@dataclass(frozen=True, slots=True)
class VehicleStart:
    """Where a vehicle's remaining work begins, and when.

    In a static study this is simply the vehicle's start stop at the beginning
    of its shift. Mid-simulation it is wherever the vehicle actually is and
    whenever it is actually free -- which is why the evaluator never has to know
    anything about execution.

    Attributes:
        stop: Where the vehicle is, or will be, when the uncommitted part of its
            route begins.
        ready_at: The earliest instant it can leave that stop.
        allow_just_in_time: Whether the vehicle may delay leaving in order to
            avoid arriving early at the first action. True for a vehicle still
            at its depot; false for one already under way, which cannot
            retroactively have left later.
        committed_timings: Recorded timings for actions the vehicle has already
            executed, one per action, from the head of the schedule. Propagation
            resumes at index ``len(committed_timings)``. Empty in a static study.
        departed_at: When the vehicle originally left its depot, for a route
            that is already under way. ``None`` means the evaluator computes it.
    """

    stop: StopId
    ready_at: Instant
    allow_just_in_time: bool = True
    committed_timings: tuple[StopTiming, ...] = ()
    departed_at: Instant | None = None


@dataclass(frozen=True, slots=True)
class EvalContext:
    """The fixed inputs to timing evaluation.

    Attributes:
        problem: The instance being solved.
        policy: How arrival turns into service and departure.
        starts: Per-vehicle overrides of where and when its remaining work
            begins. A vehicle absent from this mapping starts at its depot at
            the beginning of its shift.
        departure_tolerance: Convergence tolerance, in minutes, for the
            just-in-time depot departure solve.
        max_departure_iterations: Iteration cap for that solve.
    """

    problem: Problem
    policy: ServicePolicy = DEFAULT_POLICY
    starts: Mapping[VehicleId, VehicleStart] = field(default_factory=dict)
    departure_tolerance: float = 1e-6
    max_departure_iterations: int = 32

    def start_for(self, vehicle_id: VehicleId) -> VehicleStart:
        """Where and when this vehicle's remaining work begins."""
        override = self.starts.get(vehicle_id)
        if override is not None:
            return override
        vehicle = self.problem.vehicle(vehicle_id)
        return VehicleStart(
            stop=vehicle.start_stop,
            ready_at=vehicle.available_from,
            allow_just_in_time=True,
        )

    def with_starts(self, starts: Mapping[VehicleId, VehicleStart]) -> EvalContext:
        """A copy using a different set of vehicle start states."""
        return EvalContext(
            problem=self.problem,
            policy=self.policy,
            starts=starts,
            departure_tolerance=self.departure_tolerance,
            max_departure_iterations=self.max_departure_iterations,
        )

    def with_policy(self, policy: ServicePolicy) -> EvalContext:
        """A copy using a different service policy."""
        return EvalContext(
            problem=self.problem,
            policy=policy,
            starts=self.starts,
            departure_tolerance=self.departure_tolerance,
            max_departure_iterations=self.max_departure_iterations,
        )
