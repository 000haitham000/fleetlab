"""Vehicle availability: the shift must contain the work.

Three things are checked, and the third is one the earlier Java design could not
check at all.

* The vehicle does not leave its depot before its shift opens.
* The vehicle is **back at its end stop** before its shift closes. The Java
  original measured only the last action's departure and never modelled the
  return leg, so a route that finished on time at its final stop but could not
  get the vehicle home counted as feasible.
* The shift does not exceed any maximum duration -- drivers' hours, battery
  range expressed as time, a depot that locks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fleetlab.feasibility.violation import Violation

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fleetlab.domain.schedule import Schedule
    from fleetlab.linear import Model
    from fleetlab.mathprog.variables import PickupDeliveryVars
    from fleetlab.timing.context import EvalContext
    from fleetlab.timing.timing import RouteTiming


@dataclass(frozen=True, slots=True)
class VehicleAvailability:
    """The route must fit inside the vehicle's shift, return leg included.

    Attributes:
        tolerance: Overruns below this many minutes are ignored.
    """

    tolerance: float = 1e-9

    @property
    def name(self) -> str:
        """Identifier used in reports and penalty weights."""
        return "availability"

    def check_route(
        self,
        schedule: Schedule,
        timing: RouteTiming,
        ctx: EvalContext,
    ) -> Iterator[Violation]:
        """Report early starts, late returns and over-long shifts."""
        if schedule.is_empty:
            return
        vehicle = ctx.problem.vehicle(schedule.vehicle)

        early = vehicle.available_from - timing.depot_departure
        if early > self.tolerance:
            yield Violation(
                constraint=self.name,
                magnitude=early,
                unit="minutes",
                vehicle=schedule.vehicle,
                detail=f"leaves {early:.4g} min before the shift opens",
            )

        late = timing.return_arrival - vehicle.available_until
        if late > self.tolerance:
            yield Violation(
                constraint=self.name,
                magnitude=late,
                unit="minutes",
                vehicle=schedule.vehicle,
                detail=f"returns {late:.4g} min after the shift closes",
            )

        if vehicle.max_shift_duration is not None:
            overrun = timing.route_duration - vehicle.max_shift_duration
            if overrun > self.tolerance:
                yield Violation(
                    constraint=self.name,
                    magnitude=overrun,
                    unit="minutes",
                    vehicle=schedule.vehicle,
                    detail=f"shift runs {overrun:.4g} min over its maximum",
                )

    def to_model(
        self,
        model: Model,
        variables: PickupDeliveryVars,
        ctx: EvalContext,
    ) -> None:
        """Bound each vehicle's depot nodes by its shift, and cap the shift length."""
        nodes = variables.nodes
        for vehicle_id in ctx.problem.vehicles:
            vehicle = ctx.problem.vehicle(vehicle_id)
            start = nodes.start[vehicle_id]
            finish = nodes.finish[vehicle_id]

            model.add(
                variables.service_start[start] >= vehicle.available_from,
                f"shift_open_{vehicle_id}",
            )
            model.add(
                variables.service_start[finish] <= vehicle.available_until,
                f"shift_close_{vehicle_id}",
            )
            model.add(
                variables.service_start[finish] >= variables.service_start[start],
                f"shift_order_{vehicle_id}",
            )
            if vehicle.max_shift_duration is not None:
                model.add(
                    variables.service_start[finish] - variables.service_start[start]
                    <= vehicle.max_shift_duration,
                    f"shift_length_{vehicle_id}",
                )
