"""Problem instances.

A :class:`Problem` is everything that does not change while an algorithm runs:
the stops, the requests, the fleet, the travel matrix, the capacity space. It is
the input both lanes share -- the search lane reads it to evaluate candidate
solutions, the mathematical-programming lane reads it to emit variables and rows.

Keeping it strictly separate from :class:`~fleetlab.domain.solution.Solution` is
what makes it possible to hand the *same* instance to a metaheuristic and to a
solver and compare the answers, which is the point of the framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fleetlab.domain.action import Action
from fleetlab.domain.capacity import Capacity, CapacitySpace
from fleetlab.domain.ids import RequestId, StopId, VehicleId
from fleetlab.domain.request import ActionType, Request
from fleetlab.domain.stop import Stop
from fleetlab.domain.units import HORIZON_INFINITY, Epoch, Instant
from fleetlab.domain.vehicle import Vehicle

if TYPE_CHECKING:
    from collections.abc import Iterable

    from fleetlab.od.base import ODMatrix


@dataclass(frozen=True, slots=True, eq=False)
class Problem:
    """One instance: what is to be moved, by what, between where.

    Identity-compared rather than value-compared: an instance is a large,
    unique object, and two separately built instances with the same contents are
    not interchangeable for caching purposes.

    Attributes:
        name: Identifier for the instance, used in run logs and result tables.
        capacity_space: The dimension names shared by every capacity vector here.
        stops: Every location, by id.
        requests: Every request, by id.
        vehicles: Every vehicle, by id.
        od: The travel matrix.
        epoch: Wall-clock origin, for rendering only. The core never uses it.
        horizon_end: Latest instant the study models. Used as a safe big-M and
            to bound bucket enumeration.
    """

    name: str
    capacity_space: CapacitySpace
    stops: dict[StopId, Stop]
    requests: dict[RequestId, Request]
    vehicles: dict[VehicleId, Vehicle]
    od: ODMatrix
    epoch: Epoch | None = None
    horizon_end: Instant = HORIZON_INFINITY
    _actions: dict[RequestId, tuple[Action, Action]] = field(
        default_factory=dict, compare=False, repr=False, init=False
    )

    def __post_init__(self) -> None:
        arity = len(self.capacity_space)
        for vehicle in self.vehicles.values():
            if len(vehicle.capacity) != arity:
                msg = (
                    f"Vehicle {vehicle.id!r} has a {len(vehicle.capacity)}-dimensional "
                    f"capacity but the instance space has {arity} dimensions."
                )
                raise ValueError(msg)
        cache = {
            request_id: (
                Action(request_id, ActionType.PICKUP, request.origin),
                Action(request_id, ActionType.DROPOFF, request.destination),
            )
            for request_id, request in self.requests.items()
        }
        object.__setattr__(self, "_actions", cache)

    # ---------------------------------------------------------------- lookups

    @property
    def arity(self) -> int:
        """Number of capacity dimensions."""
        return len(self.capacity_space)

    def request(self, request_id: RequestId) -> Request:
        """Look up a request.

        Raises:
            KeyError: If the id is unknown. A programmer error.
        """
        try:
            return self.requests[request_id]
        except KeyError:
            msg = f"Unknown request {request_id!r} in instance {self.name!r}."
            raise KeyError(msg) from None

    def vehicle(self, vehicle_id: VehicleId) -> Vehicle:
        """Look up a vehicle.

        Raises:
            KeyError: If the id is unknown. A programmer error.
        """
        try:
            return self.vehicles[vehicle_id]
        except KeyError:
            msg = f"Unknown vehicle {vehicle_id!r} in instance {self.name!r}."
            raise KeyError(msg) from None

    def stop(self, stop_id: StopId) -> Stop:
        """Look up a stop.

        Raises:
            KeyError: If the id is unknown. A programmer error.
        """
        try:
            return self.stops[stop_id]
        except KeyError:
            msg = f"Unknown stop {stop_id!r} in instance {self.name!r}."
            raise KeyError(msg) from None

    # ---------------------------------------------------------------- actions

    def actions_for(self, request_id: RequestId) -> tuple[Action, Action]:
        """The canonical ``(pickup, dropoff)`` action pair for a request.

        Cached on construction so that insertion heuristics, which build these
        constantly, allocate nothing.
        """
        try:
            return self._actions[request_id]
        except KeyError:
            msg = f"Unknown request {request_id!r} in instance {self.name!r}."
            raise KeyError(msg) from None

    def pickup_of(self, request_id: RequestId) -> Action:
        """The pickup action for a request."""
        return self.actions_for(request_id)[0]

    def dropoff_of(self, request_id: RequestId) -> Action:
        """The dropoff action for a request."""
        return self.actions_for(request_id)[1]

    # ----------------------------------------------------------- demand & time

    def demand_of(self, request_id: RequestId) -> Capacity:
        """Capacity consumed by a request while onboard."""
        return self.request(request_id).demand(self.arity)

    def direct_duration(self, request_id: RequestId, depart_at: Instant | None = None) -> float:
        """Travel time from a request's origin straight to its destination.

        This is the reference against which excess onboard time is measured.
        """
        request = self.request(request_id)
        fallback = request.target_time(ActionType.PICKUP) or 0.0
        when = depart_at if depart_at is not None else fallback
        return self.od.duration(when, request.origin, request.destination)

    # ------------------------------------------------------------ construction

    @classmethod
    def build(
        cls,
        name: str,
        capacity_space: CapacitySpace,
        stops: Iterable[Stop],
        requests: Iterable[Request],
        vehicles: Iterable[Vehicle],
        od: ODMatrix,
        *,
        epoch: Epoch | None = None,
        horizon_end: Instant = HORIZON_INFINITY,
    ) -> Problem:
        """Assemble an instance from iterables, indexing them by id.

        Raises:
            ValueError: On duplicate ids, or on a request referring to an
                unknown stop.
        """
        stop_index: dict[StopId, Stop] = {}
        for stop in stops:
            if stop.id in stop_index:
                msg = f"Duplicate stop id {stop.id!r}."
                raise ValueError(msg)
            stop_index[stop.id] = stop

        request_index: dict[RequestId, Request] = {}
        for request in requests:
            if request.id in request_index:
                msg = f"Duplicate request id {request.id!r}."
                raise ValueError(msg)
            request_index[request.id] = request

        vehicle_index: dict[VehicleId, Vehicle] = {}
        for vehicle in vehicles:
            if vehicle.id in vehicle_index:
                msg = f"Duplicate vehicle id {vehicle.id!r}."
                raise ValueError(msg)
            vehicle_index[vehicle.id] = vehicle

        for request in request_index.values():
            for stop_id in (request.origin, request.destination):
                if stop_id not in stop_index:
                    msg = f"Request {request.id!r} refers to unknown stop {stop_id!r}."
                    raise ValueError(msg)
        for vehicle in vehicle_index.values():
            for stop_id in (vehicle.start_stop, vehicle.end_stop):
                if stop_id not in stop_index:
                    msg = f"Vehicle {vehicle.id!r} refers to unknown stop {stop_id!r}."
                    raise ValueError(msg)

        return cls(
            name=name,
            capacity_space=capacity_space,
            stops=stop_index,
            requests=request_index,
            vehicles=vehicle_index,
            od=od,
            epoch=epoch,
            horizon_end=horizon_end,
        )

    def __str__(self) -> str:
        return (
            f"Problem({self.name!r}: {len(self.requests)} requests, "
            f"{len(self.vehicles)} vehicles, {len(self.stops)} stops)"
        )
