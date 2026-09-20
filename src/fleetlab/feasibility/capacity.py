"""Capacity: the vehicle must be able to carry what is aboard.

Because the platform moves people and goods, capacity is a **vector**, and this
constraint checks every dimension independently. A minibus can be full of
passengers and still have room for a parcel; a van can be at its payload mass
limit with load volume to spare. Collapsing that into one number would force
every mixed study into a fudge factor.

The violation magnitude is the worst per-dimension overflow, reported in that
dimension's own unit -- "3 seats over" or "48 kg over" -- so a penalty function
can weight a mass overflow differently from a seat overflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fleetlab.feasibility.violation import Violation
from fleetlab.linear import LinExpr

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fleetlab.domain.schedule import Schedule
    from fleetlab.linear import Model
    from fleetlab.mathprog.variables import PickupDeliveryVars
    from fleetlab.timing.context import EvalContext
    from fleetlab.timing.timing import RouteTiming


@dataclass(frozen=True, slots=True)
class CapacityLimit:
    """Load carried must stay within the vehicle's limit, dimension by dimension.

    Attributes:
        report_every_position: When true, one violation per offending position;
            when false, only the worst position per dimension. The compact form
            keeps penalty totals from being dominated by a single long overload
            simply because it spans many stops.
    """

    report_every_position: bool = False

    @property
    def name(self) -> str:
        """Identifier used in reports and penalty weights."""
        return "capacity"

    def check_route(
        self,
        schedule: Schedule,
        timing: RouteTiming,
        ctx: EvalContext,
    ) -> Iterator[Violation]:
        """Report per-dimension overflow along the route's load profile."""
        vehicle = ctx.problem.vehicle(schedule.vehicle)
        space = ctx.problem.capacity_space
        limit = vehicle.capacity

        worst: dict[int, tuple[float, int]] = {}

        for position, load in enumerate(timing.loads):
            overflow = load.overflow(limit)
            if overflow.is_zero():
                continue
            for dimension, amount in enumerate(overflow.values):
                if amount <= 0.0:
                    continue
                if self.report_every_position:
                    yield self._violation(schedule, space.dimensions[dimension], amount, position)
                else:
                    current = worst.get(dimension)
                    if current is None or amount > current[0]:
                        worst[dimension] = (amount, position)

        if not self.report_every_position:
            for dimension, (amount, position) in sorted(worst.items()):
                yield self._violation(schedule, space.dimensions[dimension], amount, position)

    def _violation(
        self,
        schedule: Schedule,
        dimension: str,
        amount: float,
        position: int,
    ) -> Violation:
        return Violation(
            constraint=self.name,
            magnitude=amount,
            unit=dimension,
            vehicle=schedule.vehicle,
            index=position,
            detail=f"{amount:.4g} {dimension} above the vehicle limit",
        )

    def to_model(
        self,
        model: Model,
        variables: PickupDeliveryVars,
        ctx: EvalContext,
    ) -> None:
        """Bound the load at each node by the capacity of whichever vehicle visits it.

        Two families of rows per dimension:

        * **propagation** -- crossing an arc adds that node's signed demand,
          relaxed by big-M when the arc is not taken;
        * **limit** -- the load at a node cannot exceed the capacity of the
          vehicle that visits it. Written as a sum over vehicles weighted by
          their visit variables, which keeps it linear for a heterogeneous fleet.
        """
        nodes = variables.nodes
        arity = ctx.problem.arity

        for (tail, head, vehicle_id), arc in variables.arc.items():
            for dimension in range(arity):
                added = variables.signed_demand[head][dimension]
                big_m = (
                    variables.load_big_m[dimension]
                    if dimension < len(variables.load_big_m)
                    else variables.big_m
                )
                model.add(
                    variables.load[(head, dimension)]
                    >= variables.load[(tail, dimension)] + added - big_m * (1 - arc),
                    f"load_lo_{tail}_{head}_{vehicle_id}_{dimension}",
                )
                model.add(
                    variables.load[(head, dimension)]
                    <= variables.load[(tail, dimension)] + added + big_m * (1 - arc),
                    f"load_hi_{tail}_{head}_{vehicle_id}_{dimension}",
                )

        for node in nodes.request_nodes():
            for dimension in range(arity):
                allowance = LinExpr.sum(
                    ctx.problem.vehicle(vehicle_id).capacity[dimension] * variable
                    for vehicle_id in ctx.problem.vehicles
                    for variable in variables.out_of(node, vehicle_id)
                )
                model.add(
                    variables.load[(node, dimension)] <= allowance,
                    f"capacity_{node}_{dimension}",
                )

        for vehicle_id in ctx.problem.vehicles:
            for depot in (nodes.start[vehicle_id], nodes.finish[vehicle_id]):
                for dimension in range(arity):
                    model.add(
                        variables.load[(depot, dimension)] <= 0.0,
                        f"empty_depot_{depot}_{dimension}",
                    )
