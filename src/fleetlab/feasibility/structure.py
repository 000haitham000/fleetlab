"""Structural rules: pairing and precedence.

These are the two rules that make a route a *pickup-and-delivery* route rather
than an arbitrary tour. They hold whether the loadables are passengers or
pallets.

:class:`~fleetlab.domain.schedule.Schedule` already refuses to build a pair in
the wrong order through ``with_pair_inserted``, so in normal operation these
rules never fire. They exist because a route can also be assembled by other
means -- a warm start read from a file, a solver's answer converted back, a
hand-written test fixture -- and a silent structural break is far more expensive
to debug later than a reported one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fleetlab.domain.request import ActionType
from fleetlab.feasibility.violation import Violation
from fleetlab.linear import LinExpr, equals

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fleetlab.domain.schedule import Schedule
    from fleetlab.linear import Model
    from fleetlab.mathprog.variables import PickupDeliveryVars
    from fleetlab.timing.context import EvalContext
    from fleetlab.timing.timing import RouteTiming


@dataclass(frozen=True, slots=True)
class Pairing:
    """Both ends of a request must be served, by the same vehicle.

    A route holding a pickup whose dropoff is elsewhere describes a vehicle that
    collects a loadable and never delivers it.
    """

    @property
    def name(self) -> str:
        """Identifier used in reports and penalty weights."""
        return "pairing"

    def check_route(
        self,
        schedule: Schedule,
        timing: RouteTiming,
        ctx: EvalContext,
    ) -> Iterator[Violation]:
        """Report each request with only one end on this route."""
        del timing, ctx
        seen: dict[str, set[ActionType]] = {}
        for action in schedule.actions:
            seen.setdefault(action.request, set()).add(action.type)
        for request_id, kinds in seen.items():
            if len(kinds) == 2:
                continue
            present = next(iter(kinds))
            yield Violation(
                constraint=self.name,
                magnitude=1.0,
                unit="actions",
                vehicle=schedule.vehicle,
                request=request_id,  # type: ignore[arg-type]
                detail=f"only the {present} is on this route",
            )

    def to_model(
        self,
        model: Model,
        variables: PickupDeliveryVars,
        ctx: EvalContext,
    ) -> None:
        """Each request served exactly as often as it is 'served', by one vehicle.

        Two families of rows:

        * every pickup is left exactly ``served[r]`` times across the fleet;
        * whichever vehicle leaves the pickup also leaves the dropoff.
        """
        nodes = variables.nodes
        vehicles = tuple(ctx.problem.vehicles)

        for request_id in ctx.problem.requests:
            pickup = nodes.pickup[request_id]
            dropoff = nodes.dropoff[request_id]

            departures = LinExpr.sum(
                variable
                for vehicle_id in vehicles
                for variable in variables.out_of(pickup, vehicle_id)
            )
            model.add(
                equals(departures, LinExpr.of(variables.served[request_id])),
                f"serve_once_{request_id}",
            )

            deliveries = LinExpr.sum(
                variable
                for vehicle_id in vehicles
                for variable in variables.out_of(dropoff, vehicle_id)
            )
            model.add(
                equals(deliveries, LinExpr.of(variables.served[request_id])),
                f"deliver_once_{request_id}",
            )

            for vehicle_id in vehicles:
                same_vehicle = LinExpr.sum(variables.out_of(pickup, vehicle_id)) - LinExpr.sum(
                    variables.out_of(dropoff, vehicle_id)
                )
                model.add(
                    equals(same_vehicle, 0.0),
                    f"same_vehicle_{request_id}_{vehicle_id}",
                )


@dataclass(frozen=True, slots=True)
class Precedence:
    """A request's pickup must be served before its dropoff.

    In the search lane this is a position comparison. In the model it is a time
    ordering with the direct travel leg included, which is both the correct
    statement and a useful strengthening: it forbids a dropoff scheduled
    physically sooner than the vehicle could possibly get there.
    """

    @property
    def name(self) -> str:
        """Identifier used in reports and penalty weights."""
        return "precedence"

    def check_route(
        self,
        schedule: Schedule,
        timing: RouteTiming,
        ctx: EvalContext,
    ) -> Iterator[Violation]:
        """Report each request whose dropoff precedes its pickup."""
        del ctx
        for request_id, (pickup, dropoff) in timing.request_positions.items():
            if pickup < dropoff:
                continue
            yield Violation(
                constraint=self.name,
                magnitude=float(pickup - dropoff + 1),
                unit="positions",
                vehicle=schedule.vehicle,
                request=request_id,
                index=dropoff,
                detail=f"dropoff at #{dropoff} precedes pickup at #{pickup}",
            )

    def to_model(
        self,
        model: Model,
        variables: PickupDeliveryVars,
        ctx: EvalContext,
    ) -> None:
        """Service at the dropoff starts no earlier than pickup plus dwell plus travel."""
        nodes = variables.nodes
        for request_id in ctx.problem.requests:
            pickup = nodes.pickup[request_id]
            dropoff = nodes.dropoff[request_id]
            leg = variables.travel[(pickup, dropoff)]
            model.add(
                variables.service_start[dropoff]
                >= variables.service_start[pickup] + variables.service[pickup] + leg,
                f"precedence_{request_id}",
            )
