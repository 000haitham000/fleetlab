"""Transport requests.

A request moves one or more :class:`~fleetlab.domain.loadable.Loadable` objects
from an origin stop to a destination stop. Every request is a *pair*: one pickup
and one dropoff, which is the structure shared by people-moving and
goods-moving alike. Depot-origin distribution is expressible without a special
case -- those are simply requests whose origin happens to be the depot stop.

One rule, defined once here and used everywhere
-----------------------------------------------
"The time this action is aiming at" can be defined more than one way: the
requested time, the promised time, or something derived from both. If different
code paths pick different definitions, the same schedule can be judged feasible
by one and infeasible by another -- a bug that is easy to introduce and very
hard to see.

:meth:`Request.target_time` is the single definition: **the promised time if one
has been given, otherwise the requested time**. Nothing else in the framework is
allowed to form its own answer to that question.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fleetlab.domain.capacity import Capacity, sum_capacities
from fleetlab.domain.ids import RequestId, StopId
from fleetlab.domain.units import Duration, Instant
from fleetlab.domain.window import TimeWindow

if TYPE_CHECKING:
    from fleetlab.domain.loadable import Loadable


class ActionType(enum.Enum):
    """Which end of a request an action serves."""

    PICKUP = "pickup"
    DROPOFF = "dropoff"

    @property
    def other(self) -> ActionType:
        """The opposite end of the same request."""
        return ActionType.DROPOFF if self is ActionType.PICKUP else ActionType.PICKUP

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Request:
    """A paired pickup-and-dropoff transport requirement.

    Attributes:
        id: Stable identifier, unique within an instance.
        origin: Stop where the loadables are collected.
        destination: Stop where the loadables are delivered.
        loadables: What is being moved. May be empty only in synthetic tests.
        pickup_window: Absolute window during which pickup service may begin.
        dropoff_window: Absolute window during which dropoff service may begin.
        requested_pickup: The instant the requester asked for at the origin, if any.
        requested_dropoff: The instant the requester asked for at the destination, if any.
        promised_pickup: The instant committed back to the requester, if any.
            Overrides ``requested_pickup`` wherever a target time is needed.
        promised_dropoff: As above, for the destination.
        pickup_service_duration: Dwell at the origin (boarding, loading, paperwork).
        dropoff_service_duration: Dwell at the destination (alighting, unloading).
        max_onboard_time: Optional cap on the time between leaving the origin and
            beginning service at the destination. This is the constraint that
            turns a general pickup-and-delivery instance into a dial-a-ride one;
            leave it ``None`` for goods movement where it does not apply.
        revenue: Value of serving this request. Used by objectives that may
            leave requests unassigned.
        priority: Higher means more important. Used by insertion heuristics for
            ordering; carries no feasibility meaning.
        released_at: The instant the request becomes known to the operator. In a
            static study this is ``0.0``; in a dynamic simulation it is what
            makes the request appear mid-horizon.
        attributes: Free-form study-specific metadata. The core never reads it.
    """

    id: RequestId
    origin: StopId
    destination: StopId
    loadables: tuple[Loadable, ...]
    pickup_window: TimeWindow
    dropoff_window: TimeWindow
    requested_pickup: Instant | None = None
    requested_dropoff: Instant | None = None
    promised_pickup: Instant | None = None
    promised_dropoff: Instant | None = None
    pickup_service_duration: Duration = 0.0
    dropoff_service_duration: Duration = 0.0
    max_onboard_time: Duration | None = None
    revenue: float = 0.0
    priority: float = 0.0
    released_at: Instant = 0.0
    attributes: tuple[tuple[str, str], ...] = field(default=(), compare=False)

    # -------------------------------------------------------------- geometry

    def stop_for(self, action_type: ActionType) -> StopId:
        """The stop at which the given end of this request is served."""
        return self.origin if action_type is ActionType.PICKUP else self.destination

    def window_for(self, action_type: ActionType) -> TimeWindow:
        """The absolute service window for the given end of this request."""
        return self.pickup_window if action_type is ActionType.PICKUP else self.dropoff_window

    def service_duration_for(self, action_type: ActionType) -> Duration:
        """The dwell required at the given end of this request."""
        return (
            self.pickup_service_duration
            if action_type is ActionType.PICKUP
            else self.dropoff_service_duration
        )

    # ------------------------------------------------------------ target time

    def target_time(self, action_type: ActionType) -> Instant | None:
        """The instant this end of the request is aiming at.

        The promised time if one has been given, otherwise the requested time,
        otherwise ``None`` when the request expresses no preference beyond its
        window.

        This is the framework's *only* definition of a target time. See the
        module docstring for why that matters.
        """
        if action_type is ActionType.PICKUP:
            promised, requested = self.promised_pickup, self.requested_pickup
        else:
            promised, requested = self.promised_dropoff, self.requested_dropoff
        return promised if promised is not None else requested

    def has_commitment(self, action_type: ActionType) -> bool:
        """Whether a time has been promised back to the requester for this end."""
        if action_type is ActionType.PICKUP:
            return self.promised_pickup is not None
        return self.promised_dropoff is not None

    # ---------------------------------------------------------------- demand

    def demand(self, arity: int) -> Capacity:
        """Total capacity consumed by this request's loadables while onboard.

        Args:
            arity: Number of dimensions in the instance's capacity space, used
                to produce a correctly shaped zero when there are no loadables.
        """
        return sum_capacities([loadable.demand for loadable in self.loadables], arity)

    def signed_demand(self, action_type: ActionType, arity: int) -> Capacity:
        """Demand added to the vehicle at this end: positive at pickup, negative at dropoff."""
        demand = self.demand(arity)
        return demand if action_type is ActionType.PICKUP else demand * -1.0

    def __str__(self) -> str:
        return f"request {self.id}"
