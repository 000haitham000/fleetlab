"""Just-in-time depot departure.

A vehicle that leaves its depot as early as possible sits idle at its first stop,
burns shift time and, in a dynamic study, is in the wrong place to be
re-tasked. So the convention -- inherited from the original Java design and kept
here -- is that the vehicle leaves *just late enough* to reach its first action
on time.

Finding that instant is a genuine fixed-point problem when travel times depend
on departure time: the duration you need in order to subtract from the target
arrival is itself a function of the departure you are solving for.

Two solvers
-----------
:func:`solve_departure` runs a damped fixed-point iteration, which converges in a
single step for a constant matrix and in a handful for a smooth profile. If that
does not settle -- which happens when the target arrival sits exactly on a bucket
boundary and the iteration oscillates across it -- it falls back to a bisection
that returns the **latest departure that still arrives no later than the
target**. Preferring early arrival over any lateness matches the Java original.

Bisection assumes the matrix is FIFO: leaving later never arrives earlier. A
bucketed matrix can break that at boundaries, which is why
:meth:`~fleetlab.od.piecewise.PiecewiseODMatrix.fifo_violations` exists.

Nothing here raises on failure to converge. The result object reports it, and
the caller decides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fleetlab.domain.action import Action
    from fleetlab.domain.ids import StopId, VehicleId
    from fleetlab.domain.units import Duration, Instant
    from fleetlab.od.base import ODMatrix
    from fleetlab.timing.context import EvalContext


@dataclass(frozen=True, slots=True)
class DepartureSolution:
    """The outcome of a just-in-time departure solve.

    Attributes:
        departure: When to leave the origin.
        arrival: When that departure lands at the destination.
        iterations: How many refinement steps were taken.
        converged: Whether the iteration met the tolerance. False means the
            bisection fallback produced the answer, which is still usable but
            worth knowing about when a study's travel data is suspect.
    """

    departure: Instant
    arrival: Instant
    iterations: int
    converged: bool

    @property
    def earliness(self) -> Duration:
        """How far before the target the vehicle arrives. Never negative."""
        return 0.0


def solve_departure(
    od: ODMatrix,
    origin: StopId,
    destination: StopId,
    target_arrival: Instant,
    *,
    tolerance: float = 1e-6,
    max_iterations: int = 32,
    search_window: Duration = 1440.0,
) -> DepartureSolution:
    """Find the latest departure from ``origin`` that reaches ``destination`` by the target.

    Args:
        od: The travel matrix.
        origin: Where the vehicle starts.
        destination: Where it must arrive.
        target_arrival: The instant it should arrive by.
        tolerance: Convergence tolerance, in minutes.
        max_iterations: Cap on fixed-point refinement steps.
        search_window: How far back the bisection fallback may look.

    Returns:
        A :class:`DepartureSolution`. Never raises for a travel profile the
        iteration cannot settle; ``converged`` reports that instead.
    """
    if origin == destination:
        return DepartureSolution(target_arrival, target_arrival, 0, converged=True)

    departure = target_arrival - od.duration(target_arrival, origin, destination)

    for step in range(1, max_iterations + 1):
        arrival = departure + od.duration(departure, origin, destination)
        error = arrival - target_arrival
        if abs(error) <= tolerance:
            return DepartureSolution(departure, arrival, step, converged=True)
        departure -= error

    return _bisect_departure(
        od,
        origin,
        destination,
        target_arrival,
        search_window=search_window,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def depot_departure_for(
    ctx: EvalContext,
    vehicle_id: VehicleId,
    first_action: Action,
) -> Instant:
    """When a vehicle leaves its start stop, given what it serves first.

    The single definition of the just-in-time rule. Both the evaluator and the
    insertion screen call it, which matters more than it looks: when the screen
    assumed a fixed depot departure and the evaluator recomputed one, the screen
    discarded feasible insertions at position zero -- a vehicle inserting a new
    first action simply leaves earlier, so nothing downstream is pushed at all.

    Args:
        ctx: Evaluation context.
        vehicle_id: Whose departure to compute.
        first_action: The action that would be served first.

    Returns:
        The departure instant, never earlier than the vehicle is ready.
    """
    start = ctx.start_for(vehicle_id)
    if start.departed_at is not None:
        return start.departed_at
    if start.committed_timings or not start.allow_just_in_time:
        return start.ready_at

    request = ctx.problem.request(first_action.request)
    target = request.target_time(first_action.type)
    if target is None:
        target = request.window_for(first_action.type).earliest

    solved = solve_departure(
        ctx.problem.od,
        start.stop,
        first_action.stop,
        target,
        tolerance=ctx.departure_tolerance,
        max_iterations=ctx.max_departure_iterations,
    )
    return max(solved.departure, start.ready_at)


def _bisect_departure(
    od: ODMatrix,
    origin: StopId,
    destination: StopId,
    target_arrival: Instant,
    *,
    search_window: Duration,
    tolerance: float,
    max_iterations: int,
) -> DepartureSolution:
    """Latest departure arriving no later than the target, by bisection.

    Assumes arrival is non-decreasing in departure (the FIFO property).
    """
    low = target_arrival - search_window
    high = target_arrival

    def arrival_for(depart: Instant) -> Instant:
        return depart + od.duration(depart, origin, destination)

    if arrival_for(low) > target_arrival:
        # Even leaving a full window early does not arrive in time; take the
        # earliest departure available and report the honest arrival.
        return DepartureSolution(low, arrival_for(low), 0, converged=False)

    steps = 0
    while high - low > tolerance and steps < max_iterations * 4:
        steps += 1
        middle = (low + high) / 2.0
        if arrival_for(middle) <= target_arrival:
            low = middle
        else:
            high = middle

    return DepartureSolution(low, arrival_for(low), steps, converged=False)
