"""Time windows: service must begin inside the agreed interval.

The magnitude is minutes of lateness, which is the quantity a soft-time-window
objective penalises and the quantity a repair operator needs in order to know
which action to move first.

Earliness is handled by the service policy rather than by this constraint. A
vehicle that arrives before the window opens waits; it does not violate
anything. Whether it is *allowed* to serve early at a dropoff is a policy
decision (:class:`~fleetlab.timing.policy.EarlyArrivalPolicy` versus
:class:`~fleetlab.timing.policy.PunctualPolicy`), and the policy has already
applied it by the time the timing reaches this constraint -- which is why a
single ``B_i >= e_i`` row is the faithful model counterpart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fleetlab.domain.request import ActionType
from fleetlab.feasibility.violation import Violation

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fleetlab.domain.schedule import Schedule
    from fleetlab.linear import Model
    from fleetlab.mathprog.variables import PickupDeliveryVars
    from fleetlab.timing.context import EvalContext
    from fleetlab.timing.timing import RouteTiming


@dataclass(frozen=True, slots=True)
class TimeWindows:
    """Service at every action must start within that action's window.

    Attributes:
        tolerance: Lateness below this many minutes is ignored, to keep floating
            point noise out of violation reports.
    """

    tolerance: float = 1e-9

    @property
    def name(self) -> str:
        """Identifier used in reports and penalty weights."""
        return "time_window"

    def check_route(
        self,
        schedule: Schedule,
        timing: RouteTiming,
        ctx: EvalContext,
    ) -> Iterator[Violation]:
        """Report minutes of lateness at each action that starts too late."""
        for stop in timing.stops:
            action = schedule.actions[stop.index]
            window = ctx.problem.request(action.request).window_for(action.type)
            lateness = window.lateness(stop.service_start)
            if lateness <= self.tolerance:
                continue
            yield Violation(
                constraint=self.name,
                magnitude=lateness,
                unit="minutes",
                vehicle=schedule.vehicle,
                request=action.request,
                index=stop.index,
                detail=f"service starts {lateness:.4g} min after {window} closes",
            )

    def to_model(
        self,
        model: Model,
        variables: PickupDeliveryVars,
        ctx: EvalContext,
    ) -> None:
        """Bound each request node's service start by its window."""
        nodes = variables.nodes
        for request_id in ctx.problem.requests:
            request = ctx.problem.request(request_id)
            for action_type, node in (
                (ActionType.PICKUP, nodes.pickup[request_id]),
                (ActionType.DROPOFF, nodes.dropoff[request_id]),
            ):
                window = request.window_for(action_type)
                model.add(
                    variables.service_start[node] >= window.earliest,
                    f"tw_open_{node}",
                )
                model.add(
                    variables.service_start[node] <= window.latest,
                    f"tw_close_{node}",
                )
