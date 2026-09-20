"""Service policy: when service starts, and how long it takes.

The rule "a vehicle arriving early at a pickup waits, but a vehicle arriving
early at a dropoff serves immediately" is a **modelling decision**, not an
implementation detail. It changes the answer, it differs between studies, and it
decides the shape of the corresponding mathematical program -- whether you need
``B_i >= e_i`` or ``B_i = max(A_i, e_i)``. In the earlier Java design it was
hard-coded into the arrival calculation and could not be varied without editing
the evaluator.

Here it is an injected strategy, so a study can state its own convention and
have both lanes follow it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from fleetlab.domain.request import ActionType

if TYPE_CHECKING:
    from fleetlab.domain.request import Request
    from fleetlab.domain.units import Duration, Instant
    from fleetlab.domain.vehicle import Vehicle


@runtime_checkable
class ServicePolicy(Protocol):
    """How arrival at a stop turns into service start and departure."""

    def service_start(
        self,
        *,
        request: Request,
        action_type: ActionType,
        arrival: Instant,
        vehicle: Vehicle,
    ) -> Instant:
        """When service begins, given that the vehicle arrived at ``arrival``.

        Must never return a value before ``arrival``: a vehicle cannot serve a
        stop before reaching it.
        """
        ...

    def dwell(
        self,
        *,
        request: Request,
        action_type: ActionType,
        shares_stop_with_previous: bool,
    ) -> Duration:
        """How long service takes once it has begun."""
        ...


@dataclass(frozen=True, slots=True)
class EarlyArrivalPolicy:
    """The default convention, matching the original Java semantics.

    * A vehicle moves to the next stop as soon as it is done with the current
      one, so it may arrive early.
    * Arriving early at a **pickup**, it holds until the request's target time
      (promised if given, else requested, else the window's earliest). The
      justification is that the loadable may not be ready before then -- a
      passenger has not come out, a consignment has not been palletised.
    * Arriving early at a **dropoff**, it serves immediately and moves on,
      handing the slack to the next action.
    * Service that would begin inside a driver break is pushed to the end of the
      break. The Java original stored breaks but never applied them.

    Attributes:
        hold_at_dropoff: Set true for studies where a delivery cannot be made
            before its window opens -- a receiving bay that is not yet staffed,
            a recipient who is not yet home.
        shared_stop_factor: Multiplier applied to dwell when the previous action
            was at the same stop, modelling shared setup. The Java original
            hard-coded ``0.5``; the default here is ``1.0``, because halving a
            service time is a substantive assumption that a study should make
            deliberately rather than inherit.
        respect_breaks: Whether driver breaks push service later.
    """

    hold_at_dropoff: bool = False
    shared_stop_factor: float = 1.0
    respect_breaks: bool = True

    def service_start(
        self,
        *,
        request: Request,
        action_type: ActionType,
        arrival: Instant,
        vehicle: Vehicle,
    ) -> Instant:
        """When service begins. See the class docstring for the convention."""
        start = arrival
        if action_type is ActionType.PICKUP or self.hold_at_dropoff:
            target = request.target_time(action_type)
            floor = target if target is not None else request.window_for(action_type).earliest
            start = max(start, floor)
        if self.respect_breaks:
            start = _push_past_breaks(vehicle, start)
        return start

    def dwell(
        self,
        *,
        request: Request,
        action_type: ActionType,
        shares_stop_with_previous: bool,
    ) -> Duration:
        """Dwell for this action, discounted if it shares a stop with the previous one."""
        base = request.service_duration_for(action_type)
        return base * self.shared_stop_factor if shares_stop_with_previous else base


@dataclass(frozen=True, slots=True)
class PunctualPolicy:
    """A vehicle never serves outside a window, at either end.

    Useful for goods movement into scheduled receiving slots, and for studies
    that want the simpler mathematical program that a pure ``B_i >= e_i``
    formulation gives.

    Attributes:
        respect_breaks: Whether driver breaks push service later.
    """

    respect_breaks: bool = True

    def service_start(
        self,
        *,
        request: Request,
        action_type: ActionType,
        arrival: Instant,
        vehicle: Vehicle,
    ) -> Instant:
        """Service begins no earlier than the window opens."""
        start = max(arrival, request.window_for(action_type).earliest)
        if self.respect_breaks:
            start = _push_past_breaks(vehicle, start)
        return start

    def dwell(
        self,
        *,
        request: Request,
        action_type: ActionType,
        shares_stop_with_previous: bool,  # noqa: ARG002
    ) -> Duration:
        """Dwell for this action, with no shared-stop discount."""
        return request.service_duration_for(action_type)


def _push_past_breaks(vehicle: Vehicle, start: Instant) -> Instant:
    """Advance ``start`` past any driver break it would fall inside.

    Applied repeatedly so that back-to-back breaks are handled.
    """
    moved = True
    while moved:
        moved = False
        for rest in vehicle.breaks:
            if rest.start <= start < rest.end:
                start = rest.end
                moved = True
    return start


DEFAULT_POLICY: ServicePolicy = EarlyArrivalPolicy()
"""The policy used when a study does not name one."""
