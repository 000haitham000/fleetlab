"""Maximum onboard time: a cap on how long a loadable stays aboard.

This is the rule that turns a general pickup-and-delivery instance into a
dial-a-ride one. It is **optional per request**: a study moving people registers
it and sets ``max_onboard_time`` on each request; a study moving parcels usually
leaves the field ``None``, and then this constraint reports nothing even when
registered.

Keeping it optional rather than baking a ride-time limit into the core is what
lets the same framework serve both without the core ever asking what is being
carried.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fleetlab.feasibility.violation import Violation
from fleetlab.linear import equals

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fleetlab.domain.schedule import Schedule
    from fleetlab.linear import Model
    from fleetlab.mathprog.variables import PickupDeliveryVars
    from fleetlab.timing.context import EvalContext
    from fleetlab.timing.timing import RouteTiming


@dataclass(frozen=True, slots=True)
class MaxOnboardTime:
    """Time between leaving the origin and being served at the destination is capped.

    Attributes:
        tolerance: Overruns below this many minutes are ignored.
        default_limit: Cap applied to requests that do not state one. ``None``
            leaves such requests unconstrained, which is the usual choice for
            goods movement.
    """

    tolerance: float = 1e-9
    default_limit: float | None = None

    @property
    def name(self) -> str:
        """Identifier used in reports and penalty weights."""
        return "max_onboard_time"

    def limit_for(self, request_max: float | None) -> float | None:
        """Resolve a request's cap, falling back to :attr:`default_limit`."""
        return request_max if request_max is not None else self.default_limit

    def check_route(
        self,
        schedule: Schedule,
        timing: RouteTiming,
        ctx: EvalContext,
    ) -> Iterator[Violation]:
        """Report minutes aboard above each request's cap."""
        for request_id in timing.request_positions:
            limit = self.limit_for(ctx.problem.request(request_id).max_onboard_time)
            if limit is None:
                continue
            onboard = timing.onboard_time(request_id)
            if onboard is None:
                continue
            overrun = onboard - limit
            if overrun <= self.tolerance:
                continue
            yield Violation(
                constraint=self.name,
                magnitude=overrun,
                unit="minutes",
                vehicle=schedule.vehicle,
                request=request_id,
                detail=f"{onboard:.4g} min aboard against a {limit:.4g} min limit",
            )

    def to_model(
        self,
        model: Model,
        variables: PickupDeliveryVars,
        ctx: EvalContext,
    ) -> None:
        """Define the onboard-time variable and bound it above and below.

        The lower bound is the direct origin-to-destination travel time, which is
        a valid strengthening: no routing can deliver faster than driving
        straight there.
        """
        nodes = variables.nodes
        for request_id in ctx.problem.requests:
            pickup = nodes.pickup[request_id]
            dropoff = nodes.dropoff[request_id]
            onboard = variables.onboard[request_id]

            model.add(
                equals(
                    onboard
                    - variables.service_start[dropoff]
                    + variables.service_start[pickup]
                    + variables.service[pickup],
                    0.0,
                ),
                f"onboard_def_{request_id}",
            )
            model.add(
                onboard >= variables.travel[(pickup, dropoff)],
                f"onboard_lo_{request_id}",
            )
            limit = self.limit_for(ctx.problem.request(request_id).max_onboard_time)
            if limit is not None:
                model.add(onboard <= limit, f"onboard_hi_{request_id}")
