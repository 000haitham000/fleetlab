"""Locking: how much of a plan is too imminent to change.

A dynamic optimiser that is free to rewrite the whole plan at every decision
epoch will send a vehicle towards a stop, change its mind, and send it
somewhere else -- repeatedly, while the vehicle is already driving. Locking is
what stops that.

This generalises the rule from the earlier Java design, which locked only the
single next action and computed its horizon as the larger of a fixed window and
a fraction of the leg leading to it. Both parts are kept -- the fixed window
handles short legs, the fraction handles long ones -- but the policy is now an
injected strategy returning a committed **prefix length**, so a study can lock
more than one action, lock by distance instead of time, or not lock at all.

The result feeds :attr:`~fleetlab.domain.schedule.Schedule.committed`, which
turns "may I insert here?" into an integer comparison rather than a scan for a
started action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fleetlab.domain.schedule import Schedule
    from fleetlab.domain.units import Duration, Instant
    from fleetlab.timing.context import EvalContext
    from fleetlab.timing.timing import RouteTiming


@runtime_checkable
class LockingPolicy(Protocol):
    """Decides how much of a route is frozen against re-planning."""

    @property
    def name(self) -> str:
        """Identifier for run logs."""
        ...

    def committed_prefix(
        self,
        schedule: Schedule,
        timing: RouteTiming,
        ctx: EvalContext,
        now: Instant,
        executed: int,
    ) -> int:
        """How many leading actions may not be reordered.

        Must return at least ``executed``: an action already performed cannot
        become unfrozen.
        """
        ...


@dataclass(frozen=True, slots=True)
class HorizonLocking:
    """Freeze actions whose service is imminent.

    An action is frozen when its service would begin within
    ``max(window, fraction x leg)`` of now, where the leg is the time since the
    previous action's departure. The fixed window covers short hops where a
    fraction would be meaninglessly small; the fraction covers long legs where a
    fixed window would leave a vehicle re-tasked most of the way to its
    destination.

    Attributes:
        window: Minimum locking horizon.
        fraction: Share of the incoming leg that also counts as locked.
        max_actions: Cap on how many actions may be frozen at once. Locking too
            much starves the optimiser; the default of one action matches the
            Java original's behaviour.
    """

    window: Duration = 5.0
    fraction: float = 0.5
    max_actions: int = 1

    @property
    def name(self) -> str:
        """Identifier for run logs."""
        return "horizon"

    def committed_prefix(
        self,
        schedule: Schedule,
        timing: RouteTiming,
        ctx: EvalContext,
        now: Instant,
        executed: int,
    ) -> int:
        """Freeze the executed prefix plus any imminent actions after it."""
        del ctx
        frozen = executed
        for position in range(executed, len(schedule.actions)):
            if frozen - executed >= self.max_actions:
                break
            stop = timing.stops[position]
            previous_departure = (
                timing.stops[position - 1].departure if position else timing.depot_departure
            )
            leg = max(0.0, stop.service_start - previous_departure)
            horizon = max(self.window, self.fraction * leg)
            if now + horizon < stop.service_start:
                break
            frozen = position + 1
        return frozen


@dataclass(frozen=True, slots=True)
class NoLocking:
    """Freeze only what has already happened.

    Correct for a study measuring the value of perfect flexibility, and a useful
    control: the gap between this and :class:`HorizonLocking` is the price of
    not thrashing the fleet.
    """

    @property
    def name(self) -> str:
        """Identifier for run logs."""
        return "none"

    def committed_prefix(
        self,
        schedule: Schedule,
        timing: RouteTiming,
        ctx: EvalContext,
        now: Instant,
        executed: int,
    ) -> int:
        """Freeze exactly the executed prefix."""
        del schedule, timing, ctx, now
        return executed


@dataclass(frozen=True, slots=True)
class LockOnboard:
    """Freeze everything up to and including the dropoff of anything aboard.

    The conservative choice for people-moving: once a passenger is in the
    vehicle, their destination is not renegotiated. For goods it is usually too
    strong, since a parcel does not mind being re-routed.
    """

    @property
    def name(self) -> str:
        """Identifier for run logs."""
        return "onboard"

    def committed_prefix(
        self,
        schedule: Schedule,
        timing: RouteTiming,
        ctx: EvalContext,
        now: Instant,
        executed: int,
    ) -> int:
        """Freeze through the last dropoff of a request already picked up."""
        del timing, ctx, now
        picked_up = {action.request for action in schedule.actions[:executed] if action.is_pickup}
        delivered = {action.request for action in schedule.actions[:executed] if action.is_dropoff}
        aboard = picked_up - delivered
        frozen = executed
        for position in range(executed, len(schedule.actions)):
            action = schedule.actions[position]
            if action.is_dropoff and action.request in aboard:
                frozen = position + 1
        return frozen
