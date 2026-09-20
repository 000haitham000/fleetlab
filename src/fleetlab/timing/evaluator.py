"""The forward pass.

One walk over a route produces its entire timing picture. This is the single
most-executed function in the framework, and the only place in it that knows how
a schedule turns into instants.

Properties that matter, and why
-------------------------------
**Total.** :func:`evaluate_route` never raises for a modelling reason. Any
schedule, however infeasible, has a timing -- it just has violations too. That
separation is what lets a metaheuristic traverse infeasible space under a
penalty, which most good pickup-and-delivery heuristics do deliberately.

**Pure.** It reads the schedule and the context and writes nothing. Two
candidates can therefore be evaluated concurrently, and the result can be cached
by the schedule's hash.

**Single pass.** Arrivals, services, departures, waits, loads, distances and
onboard times all come out of the same O(n) walk. The Java original recomputed
from index 0 per query, so asking a route for all its times was O(n^2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fleetlab.domain.capacity import Capacity
from fleetlab.timing.departure import depot_departure_for
from fleetlab.timing.timing import RouteTiming, SolutionTiming, StopTiming

if TYPE_CHECKING:
    from fleetlab.domain.ids import RequestId, VehicleId
    from fleetlab.domain.problem import Problem
    from fleetlab.domain.schedule import Schedule
    from fleetlab.domain.solution import Solution
    from fleetlab.domain.units import Instant
    from fleetlab.timing.context import EvalContext, VehicleStart


def evaluate_route(schedule: Schedule, ctx: EvalContext) -> RouteTiming:
    """Compute the complete timing of one route.

    Args:
        schedule: The route to time. Not modified.
        ctx: Instance, service policy and per-vehicle start states.

    Returns:
        A :class:`~fleetlab.timing.timing.RouteTiming` covering every action,
        the load profile, and the return leg to the vehicle's end stop.
    """
    problem = ctx.problem
    vehicle = problem.vehicle(schedule.vehicle)
    start = ctx.start_for(schedule.vehicle)
    zero = Capacity((0.0,) * problem.arity)

    if not schedule.actions:
        departed = start.departed_at if start.departed_at is not None else start.ready_at
        return RouteTiming(
            vehicle=schedule.vehicle,
            depot_departure=departed,
            stops=(),
            loads=(),
            return_arrival=departed,
            travel_distance=0.0,
            travel_time=0.0,
        )

    history = start.committed_timings
    resume_at = len(history)
    if resume_at > len(schedule.actions):
        msg = (
            f"Vehicle {schedule.vehicle!r} has {resume_at} recorded timings but its "
            f"schedule holds only {len(schedule.actions)} actions."
        )
        raise ValueError(msg)

    stops: list[StopTiming] = list(history)
    travel_time = 0.0
    travel_distance = 0.0

    here = start.stop
    now = start.ready_at

    depot_departure = _initial_departure(schedule, ctx, start, resume_at)
    if resume_at == 0:
        now = depot_departure

    for index in range(resume_at, len(schedule.actions)):
        action = schedule.actions[index]
        request = problem.request(action.request)

        leg = ctx.problem.od.duration(now, here, action.stop)
        travel_time += leg
        travel_distance += ctx.problem.od.distance(here, action.stop)
        arrival = now + leg

        service = ctx.policy.service_start(
            request=request,
            action_type=action.type,
            arrival=arrival,
            vehicle=vehicle,
        )
        shares_stop = index > 0 and schedule.actions[index - 1].stop == action.stop
        dwell = ctx.policy.dwell(
            request=request,
            action_type=action.type,
            shares_stop_with_previous=shares_stop,
        )
        departure = service + dwell

        stops.append(StopTiming(index, arrival, service, departure))
        now = departure
        here = action.stop

    return_leg = ctx.problem.od.duration(now, here, vehicle.end_stop)
    travel_time += return_leg
    travel_distance += ctx.problem.od.distance(here, vehicle.end_stop)
    return_arrival = now + return_leg

    frozen_stops = tuple(stops)
    loads = _load_profile(schedule, problem, zero)
    positions = _request_positions(schedule)
    directs = _direct_durations(schedule, ctx, frozen_stops, positions)

    return RouteTiming(
        vehicle=schedule.vehicle,
        depot_departure=depot_departure,
        stops=frozen_stops,
        loads=loads,
        return_arrival=return_arrival,
        travel_distance=travel_distance,
        travel_time=travel_time,
        request_positions=positions,
        direct_durations=directs,
    )


def evaluate_solution(solution: Solution, ctx: EvalContext) -> SolutionTiming:
    """Compute timings for every route in a solution."""
    routes: dict[VehicleId, RouteTiming] = {
        schedule.vehicle: evaluate_route(schedule, ctx) for schedule in solution.routes
    }
    return SolutionTiming(routes)


# --------------------------------------------------------------------- helpers


def _initial_departure(
    schedule: Schedule,
    ctx: EvalContext,
    start: VehicleStart,
    resume_at: int,
) -> Instant:
    """When the vehicle leaves its start stop.

    A vehicle still at its depot leaves just late enough to reach its first
    action on time, never before its shift opens. A vehicle already under way
    cannot retroactively have left later, so its recorded departure stands.

    Clamping to the shift opening means the vehicle will be late for its first
    action. That is a violation for the feasibility layer to report, not an
    error here.
    """
    if resume_at > 0:
        return start.departed_at if start.departed_at is not None else start.ready_at
    return depot_departure_for(ctx, schedule.vehicle, schedule.actions[0])


def _load_profile(
    schedule: Schedule,
    problem: Problem,
    zero: Capacity,
) -> tuple[Capacity, ...]:
    """Cumulative load carried after completing each action."""
    loads: list[Capacity] = []
    carried = zero
    for action in schedule.actions:
        request = problem.request(action.request)
        carried = carried + request.signed_demand(action.type, problem.arity)
        loads.append(carried)
    return tuple(loads)


def _request_positions(schedule: Schedule) -> dict[RequestId, tuple[int, int]]:
    """Pickup and dropoff index per request that is fully present on the route."""
    pickups: dict[RequestId, int] = {}
    dropoffs: dict[RequestId, int] = {}
    for index, action in enumerate(schedule.actions):
        target = pickups if action.is_pickup else dropoffs
        target.setdefault(action.request, index)
    return {
        request: (pickups[request], dropoffs[request]) for request in pickups if request in dropoffs
    }


def _direct_durations(
    schedule: Schedule,
    ctx: EvalContext,
    stops: tuple[StopTiming, ...],
    positions: dict[RequestId, tuple[int, int]],
) -> dict[RequestId, float]:
    """Origin-to-destination travel time per request, at its actual pickup departure.

    Taking the direct duration at the *actual* departure instant rather than at
    some nominal time matters under a time-dependent matrix: otherwise a
    detour's excess is measured against a reference the vehicle could never have
    achieved.
    """
    del schedule
    directs: dict[RequestId, float] = {}
    for request_id, (pickup_index, _) in positions.items():
        request = ctx.problem.request(request_id)
        depart_at = stops[pickup_index].departure
        directs[request_id] = ctx.problem.od.duration(
            depart_at, request.origin, request.destination
        )
    return directs


class Evaluator:
    """A memoising wrapper around :func:`evaluate_route`.

    A schedule is hashable and immutable, so its timing can be cached safely.
    This pays off heavily in ruin-and-recreate, where the same partial route is
    rebuilt and re-evaluated many times over a run.

    The cache is keyed on the schedule alone, so one evaluator belongs to one
    context. Changing policy or start states means a new evaluator.
    """

    __slots__ = ("_cache", "_ctx", "_hits", "_misses")

    def __init__(self, ctx: EvalContext) -> None:
        self._ctx = ctx
        self._cache: dict[Schedule, RouteTiming] = {}
        self._hits = 0
        self._misses = 0

    @property
    def context(self) -> EvalContext:
        """The context every evaluation uses."""
        return self._ctx

    def route(self, schedule: Schedule) -> RouteTiming:
        """Timing for one route, from cache when available."""
        cached = self._cache.get(schedule)
        if cached is not None:
            self._hits += 1
            return cached
        self._misses += 1
        timing = evaluate_route(schedule, self._ctx)
        self._cache[schedule] = timing
        return timing

    def solution(self, solution: Solution) -> SolutionTiming:
        """Timings for every route in a solution."""
        return SolutionTiming(
            {schedule.vehicle: self.route(schedule) for schedule in solution.routes}
        )

    @property
    def statistics(self) -> tuple[int, int]:
        """Cache ``(hits, misses)``. Useful when tuning a search."""
        return (self._hits, self._misses)

    def clear(self) -> None:
        """Drop the cache. Call between independent runs to bound memory."""
        self._cache.clear()
