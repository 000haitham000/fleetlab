"""The objective terms most pickup-and-delivery studies compose from.

Each states its cost twice -- once by reading a route's timing, once as a linear
expression -- for the same reason constraints do. A study whose search lane and
model lane optimise different functions is not comparing algorithms; it is
comparing problems.

Terms are grouped by whose interest they serve, because that is how the
trade-off is usually argued about:

* **operator** -- distance, time, fleet size, driver idle time;
* **loadable** -- excess onboard time, lateness against the promised window;
* **commercial** -- revenue foregone on requests left unserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fleetlab.linear import LinExpr

if TYPE_CHECKING:
    from fleetlab.domain.schedule import Schedule
    from fleetlab.domain.solution import Solution
    from fleetlab.mathprog.variables import PickupDeliveryVars
    from fleetlab.timing.context import EvalContext
    from fleetlab.timing.timing import RouteTiming, SolutionTiming


# ------------------------------------------------------------------- operator


@dataclass(frozen=True, slots=True)
class TravelDistance:
    """Total distance driven, including the return to the end depot."""

    @property
    def name(self) -> str:
        """Breakdown key."""
        return "travel_distance"

    def route_cost(self, schedule: Schedule, timing: RouteTiming, ctx: EvalContext) -> float:
        """Distance driven by one vehicle."""
        del schedule, ctx
        return timing.travel_distance

    def fleet_cost(self, solution: Solution, timings: SolutionTiming, ctx: EvalContext) -> float:
        """Nothing; this term is fully attributable to routes."""
        del solution, timings, ctx
        return 0.0

    def to_expr(self, variables: PickupDeliveryVars, ctx: EvalContext) -> LinExpr:
        """Sum of arc distances over selected arcs."""
        del ctx
        return LinExpr.sum(
            variables.distance[(tail, head)] * arc for (tail, head, _), arc in variables.arc.items()
        )


@dataclass(frozen=True, slots=True)
class TravelTime:
    """Total time spent moving, excluding waiting and service."""

    @property
    def name(self) -> str:
        """Breakdown key."""
        return "travel_time"

    def route_cost(self, schedule: Schedule, timing: RouteTiming, ctx: EvalContext) -> float:
        """Driving time for one vehicle."""
        del schedule, ctx
        return timing.travel_time

    def fleet_cost(self, solution: Solution, timings: SolutionTiming, ctx: EvalContext) -> float:
        """Nothing; this term is fully attributable to routes."""
        del solution, timings, ctx
        return 0.0

    def to_expr(self, variables: PickupDeliveryVars, ctx: EvalContext) -> LinExpr:
        """Sum of arc travel times over selected arcs."""
        del ctx
        return LinExpr.sum(
            variables.travel[(tail, head)] * arc for (tail, head, _), arc in variables.arc.items()
        )


@dataclass(frozen=True, slots=True)
class RouteDuration:
    """Total elapsed shift time across the fleet.

    Distinct from travel time: it includes waiting and service, so it is the
    term to use when drivers are paid by the hour rather than by the kilometre.
    """

    @property
    def name(self) -> str:
        """Breakdown key."""
        return "route_duration"

    def route_cost(self, schedule: Schedule, timing: RouteTiming, ctx: EvalContext) -> float:
        """Elapsed shift time for one vehicle, zero if it is unused."""
        del ctx
        return 0.0 if schedule.is_empty else timing.route_duration

    def fleet_cost(self, solution: Solution, timings: SolutionTiming, ctx: EvalContext) -> float:
        """Nothing; this term is fully attributable to routes."""
        del solution, timings, ctx
        return 0.0

    def to_expr(self, variables: PickupDeliveryVars, ctx: EvalContext) -> LinExpr:
        """Sum over vehicles of shift end minus shift start."""
        nodes = variables.nodes
        return LinExpr.sum(
            variables.service_start[nodes.finish[vehicle_id]]
            - variables.service_start[nodes.start[vehicle_id]]
            for vehicle_id in ctx.problem.vehicles
        )


@dataclass(frozen=True, slots=True)
class DriverWait:
    """Total driver idle time, from arriving somewhere before it is time to serve.

    Worth costing even when it is free: idle time is capacity the fleet is not
    using, and penalising it lightly tends to produce schedules that survive
    disruption better.
    """

    @property
    def name(self) -> str:
        """Breakdown key."""
        return "driver_wait"

    def route_cost(self, schedule: Schedule, timing: RouteTiming, ctx: EvalContext) -> float:
        """Idle time on one route."""
        del schedule, ctx
        return timing.total_wait

    def fleet_cost(self, solution: Solution, timings: SolutionTiming, ctx: EvalContext) -> float:
        """Nothing; this term is fully attributable to routes."""
        del solution, timings, ctx
        return 0.0


@dataclass(frozen=True, slots=True)
class FleetSize:
    """A fixed charge for every vehicle that is used at all.

    Set the weight high to minimise fleet size first and routing cost second,
    which is the usual hierarchy in strategic studies.
    """

    @property
    def name(self) -> str:
        """Breakdown key."""
        return "fleet_size"

    def route_cost(self, schedule: Schedule, timing: RouteTiming, ctx: EvalContext) -> float:
        """The vehicle's fixed cost if it carries anything, otherwise nothing."""
        del timing
        if schedule.is_empty:
            return 0.0
        vehicle = ctx.problem.vehicle(schedule.vehicle)
        return vehicle.fixed_cost if vehicle.fixed_cost else 1.0

    def fleet_cost(self, solution: Solution, timings: SolutionTiming, ctx: EvalContext) -> float:
        """Nothing; this term is fully attributable to routes."""
        del solution, timings, ctx
        return 0.0

    def to_expr(self, variables: PickupDeliveryVars, ctx: EvalContext) -> LinExpr:
        """Charge a vehicle whenever it leaves its depot for a request node."""
        nodes = variables.nodes
        request_nodes = frozenset(nodes.request_nodes())
        parts: list[LinExpr] = []
        for vehicle_id in ctx.problem.vehicles:
            vehicle = ctx.problem.vehicle(vehicle_id)
            charge = vehicle.fixed_cost if vehicle.fixed_cost else 1.0
            start = nodes.start[vehicle_id]
            for head in nodes.nodes_for(vehicle_id):
                key = (start, head, vehicle_id)
                if head in request_nodes and key in variables.arc:
                    parts.append(LinExpr.of(variables.arc[key]) * charge)
        return LinExpr.sum(parts)


# ------------------------------------------------------------------- loadable


@dataclass(frozen=True, slots=True)
class ExcessOnboardTime:
    """Time aboard above the direct origin-to-destination trip.

    The standard quality-of-service term for people-moving: it measures the
    detour a shared ride imposes. For goods it measures how long a consignment
    sits in a vehicle beyond necessity, which matters for perishables.
    """

    @property
    def name(self) -> str:
        """Breakdown key."""
        return "excess_onboard"

    def route_cost(self, schedule: Schedule, timing: RouteTiming, ctx: EvalContext) -> float:
        """Summed excess onboard time over requests this route serves."""
        del schedule, ctx
        return timing.total_excess_onboard_time()

    def fleet_cost(self, solution: Solution, timings: SolutionTiming, ctx: EvalContext) -> float:
        """Nothing; this term is fully attributable to routes."""
        del solution, timings, ctx
        return 0.0

    def to_expr(self, variables: PickupDeliveryVars, ctx: EvalContext) -> LinExpr:
        """Onboard time less the direct leg, summed over requests."""
        nodes = variables.nodes
        parts: list[LinExpr] = []
        for request_id in ctx.problem.requests:
            direct = variables.travel[(nodes.pickup[request_id], nodes.dropoff[request_id])]
            parts.append(LinExpr.of(variables.onboard[request_id]) - direct)
        return LinExpr.sum(parts)


@dataclass(frozen=True, slots=True)
class Lateness:
    """Minutes of service beginning after the target time.

    Distinct from the time-window *constraint*: a window is a hard bound, this
    is the soft cost of being late against what was promised. A study can use
    either, or both with a narrow hard window around a soft target.
    """

    @property
    def name(self) -> str:
        """Breakdown key."""
        return "lateness"

    def route_cost(self, schedule: Schedule, timing: RouteTiming, ctx: EvalContext) -> float:
        """Summed lateness against target times along one route."""
        total = 0.0
        for stop in timing.stops:
            action = schedule.actions[stop.index]
            request = ctx.problem.request(action.request)
            target = request.target_time(action.type)
            if target is None:
                continue
            total += max(0.0, stop.service_start - target)
        return total


# ----------------------------------------------------------------- commercial


@dataclass(frozen=True, slots=True)
class UnservedPenalty:
    """A charge for every request left in the unassigned pool.

    Attributes:
        use_revenue: When true, the charge is the request's stated revenue, so
            the objective naturally declines work that costs more than it earns.
            When false, every unserved request costs :attr:`flat_charge`, which
            is the right shape for a study that must serve everything and only
            tolerates gaps during search.
        flat_charge: The per-request charge when ``use_revenue`` is false.
    """

    use_revenue: bool = False
    flat_charge: float = 1.0

    @property
    def name(self) -> str:
        """Breakdown key."""
        return "unserved"

    def route_cost(self, schedule: Schedule, timing: RouteTiming, ctx: EvalContext) -> float:
        """Nothing; this term is fleet-level by nature."""
        del schedule, timing, ctx
        return 0.0

    def fleet_cost(self, solution: Solution, timings: SolutionTiming, ctx: EvalContext) -> float:
        """Charge for the unassigned pool."""
        del timings
        if not self.use_revenue:
            return self.flat_charge * len(solution.unassigned)
        return sum(ctx.problem.request(request).revenue for request in solution.unassigned)

    def to_expr(self, variables: PickupDeliveryVars, ctx: EvalContext) -> LinExpr:
        """Charge ``1 - served[r]`` per request."""
        parts: list[LinExpr] = []
        for request_id in ctx.problem.requests:
            charge = (
                ctx.problem.request(request_id).revenue if self.use_revenue else self.flat_charge
            )
            parts.append((1.0 - LinExpr.of(variables.served[request_id])) * charge)
        return LinExpr.sum(parts)
