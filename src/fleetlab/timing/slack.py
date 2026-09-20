"""Forward time slack: the O(1) insertion filter.

The expensive part of any insertion heuristic is asking, for each of many
candidate positions, "does putting this here break something later?" Answered
naively that is an O(n) walk per candidate, so scanning a route costs O(n^2) and
a full insertion round over the fleet costs more again.

Forward time slack (Savelsbergh 1992; used throughout Cordeau & Laporte's
dial-a-ride work) precomputes, in one backward O(n) pass, how much extra delay
each position can absorb. Each candidate is then a comparison.

This is the concrete reason the route had to become immutable. The slack array
is derived from a route's timing; in a design where testing a candidate means
mutating the route, the mutation invalidates the very array you were going to
test against. Here the base route cannot change underneath it, so the array
stays valid for the whole scan.

What this is, precisely
-----------------------
A **filter**, not a verdict. It is exact when the travel matrix is
:attr:`~fleetlab.od.base.TimeDependence.CONSTANT` and no maximum-onboard-time
constraint is active. Otherwise it is a screen: a candidate it rejects is
genuinely infeasible, and a candidate it passes must still be confirmed by a
full evaluation. Two reasons:

* **Onboard time.** Delaying position *i* stretches the onboard time of any
  request picked up before *i* and dropped after it. Accounting for that exactly
  makes the bound depend on *i* in a way the backward recursion cannot express,
  so slack here deliberately ignores it and stays optimistic -- which is the safe
  direction for a filter.
* **Time-dependent travel.** With a varying matrix, departing D later can delay
  arrival by more than D if the later departure falls in a worse traffic band.
  Delay no longer propagates one-for-one, so treat the filter as a heuristic
  screen and confirm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fleetlab.domain.units import HORIZON_INFINITY

if TYPE_CHECKING:
    from fleetlab.domain.action import Action
    from fleetlab.domain.schedule import Schedule
    from fleetlab.domain.units import Duration
    from fleetlab.timing.context import EvalContext
    from fleetlab.timing.timing import RouteTiming


@dataclass(frozen=True, slots=True)
class ForwardSlack:
    """How much delay each position of a route can absorb.

    Attributes:
        depot: Maximum delay to the depot departure that nothing later rejects.
        after: ``after[i]`` is the maximum delay to the departure from position
            ``i`` that nothing later rejects.
        waits: Driver idle time at each position, which is what absorbs delay.
        exact: Whether this slack is a precise bound rather than an optimistic
            screen. True only for a constant travel matrix with no active
            maximum-onboard-time constraint.
    """

    depot: Duration
    after: tuple[Duration, ...]
    waits: tuple[Duration, ...]
    exact: bool

    def absorbable_after(self, index: int) -> Duration:
        """Maximum delay tolerable in the departure from ``index``."""
        if index < 0:
            return self.depot
        return self.after[index]

    def can_absorb(self, index: int, delay: Duration) -> bool:
        """Whether a delay introduced at ``index`` is definitely survivable.

        A ``False`` here is conclusive: the insertion is infeasible. A ``True``
        is conclusive only when :attr:`exact`; otherwise confirm by evaluating.
        """
        return delay <= self.absorbable_after(index)


def forward_slack(
    schedule: Schedule,
    timing: RouteTiming,
    ctx: EvalContext,
) -> ForwardSlack:
    """Compute the forward time slack of a timed route in one backward pass.

    Args:
        schedule: The route.
        timing: Its timing, from :func:`~fleetlab.timing.evaluator.evaluate_route`.
        ctx: The evaluation context the timing came from.

    Returns:
        A :class:`ForwardSlack` for the route.
    """
    from fleetlab.od.base import TimeDependence

    problem = ctx.problem
    vehicle = problem.vehicle(schedule.vehicle)
    count = len(timing.stops)

    onboard_limited = any(
        problem.request(request_id).max_onboard_time is not None
        for request_id in schedule.requests()
    )
    exact = problem.od.time_dependence is TimeDependence.CONSTANT and not onboard_limited

    if count == 0:
        room = vehicle.available_until - timing.return_arrival
        return ForwardSlack(depot=room, after=(), waits=(), exact=exact)

    waits = tuple(stop.wait for stop in timing.stops)

    # Local room at a position: how much later service could begin there before
    # breaking that position's own window.
    local: list[Duration] = []
    for index, stop in enumerate(timing.stops):
        action = schedule.actions[index]
        window = problem.request(action.request).window_for(action.type)
        local.append(window.latest - stop.service_start)

    # Backward recursion. The last position's departure is bounded only by the
    # need to get the vehicle home before its shift ends.
    after: list[Duration] = [0.0] * count
    after[count - 1] = vehicle.available_until - timing.return_arrival
    for index in range(count - 2, -1, -1):
        following = index + 1
        after[index] = waits[following] + min(local[following], after[following])

    depot = waits[0] + min(local[0], after[0])

    return ForwardSlack(depot=depot, after=tuple(after), waits=waits, exact=exact)


def insertion_push(
    schedule: Schedule,
    timing: RouteTiming,
    ctx: EvalContext,
    action: Action,
    index: int,
) -> Duration:
    """Extra time a single insertion at ``index`` pushes onto everything after it.

    This is the detour cost in time: go to the new stop, serve it, carry on --
    minus the leg it replaces. Waiting the new action itself incurs is included,
    because a vehicle held at the new stop departs later.

    Inserting at position zero is a special case that is easy to get wrong.
    Under the just-in-time rule the vehicle's depot departure is not fixed: a
    new first action means it simply leaves earlier, and the rest of the route
    may not be pushed at all. Treating the old departure as fixed here made the
    screen reject feasible insertions, so the new departure is re-solved through
    the same :func:`~fleetlab.timing.departure.depot_departure_for` the
    evaluator uses.

    Args:
        schedule: The route as it stands.
        timing: Its timing.
        ctx: Evaluation context.
        action: The action to insert.
        index: Position it would occupy.

    Returns:
        Minutes of delay imposed on position ``index`` onward. Never negative.
    """
    from fleetlab.timing.departure import depot_departure_for

    problem = ctx.problem
    vehicle = problem.vehicle(schedule.vehicle)
    request = problem.request(action.request)

    if index == 0:
        previous_stop = ctx.start_for(schedule.vehicle).stop
        depart_previous = depot_departure_for(ctx, schedule.vehicle, action)
    else:
        previous_stop = schedule.actions[index - 1].stop
        depart_previous = timing.stops[index - 1].departure

    next_stop = schedule.actions[index].stop if index < len(schedule.actions) else vehicle.end_stop

    leg_in = problem.od.duration(depart_previous, previous_stop, action.stop)
    arrival = depart_previous + leg_in
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
    leg_out = problem.od.duration(departure, action.stop, next_stop)

    new_arrival_at_next = departure + leg_out

    if index == 0:
        # Compare against what the following action's arrival actually was,
        # since the departure this insertion implies may differ from the one the
        # base timing used.
        original_arrival_at_next = (
            timing.stops[0].arrival if timing.stops else timing.return_arrival
        )
    else:
        original_leg = problem.od.duration(depart_previous, previous_stop, next_stop)
        original_arrival_at_next = depart_previous + original_leg

    return max(0.0, new_arrival_at_next - original_arrival_at_next)


def route_duration_room(
    schedule: Schedule,
    timing: RouteTiming,
    ctx: EvalContext,
) -> Duration:
    """How much longer the route may become before it breaks the shift.

    Returns :data:`~fleetlab.domain.units.HORIZON_INFINITY` when the vehicle has
    no maximum shift duration and its availability end is unbounded.
    """
    vehicle = ctx.problem.vehicle(schedule.vehicle)
    room = vehicle.available_until - timing.return_arrival
    if vehicle.max_shift_duration is not None:
        room = min(room, vehicle.max_shift_duration - timing.route_duration)
    return min(room, HORIZON_INFINITY)
