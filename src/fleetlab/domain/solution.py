"""Solutions: the fleet-wide plan.

A :class:`Solution` is every vehicle's route plus the pool of requests that are
not currently served by any of them.

The unassigned pool is not an afterthought. It is what makes ruin-and-recreate
expressible at all -- a ruin operator's whole job is to move requests from routes
into the pool -- and it is the natural home for requests a study deliberately
rejects because serving them costs more than their revenue. The earlier Java
design had no fleet-level object at all, so there was nowhere for either to live.

Like :class:`~fleetlab.domain.schedule.Schedule`, this is immutable data with a
mutable editor for convenience. See that module for why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, Self

from fleetlab.domain.ids import RequestId, VehicleId
from fleetlab.domain.schedule import Schedule

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


@dataclass(frozen=True, slots=True)
class Solution:
    """A plan for the whole fleet.

    Attributes:
        routes: One schedule per vehicle, including empty ones. Keeping empty
            routes present rather than omitting them means the set of vehicles
            never has to be consulted separately to know what is available.
        unassigned: Requests not currently on any route.
    """

    routes: tuple[Schedule, ...] = ()
    unassigned: frozenset[RequestId] = frozenset()
    _by_vehicle: dict[VehicleId, Schedule] = field(
        default_factory=dict, compare=False, repr=False, hash=False, init=False
    )

    def __post_init__(self) -> None:
        index = {schedule.vehicle: schedule for schedule in self.routes}
        if len(index) != len(self.routes):
            msg = "Solution contains more than one route for the same vehicle."
            raise ValueError(msg)
        object.__setattr__(self, "_by_vehicle", index)

    # ---------------------------------------------------------- construction

    @classmethod
    def empty(cls, vehicles: Iterable[VehicleId], unassigned: Iterable[RequestId]) -> Solution:
        """A solution with an empty route per vehicle and every request unassigned."""
        return cls(
            routes=tuple(Schedule(vehicle) for vehicle in vehicles),
            unassigned=frozenset(unassigned),
        )

    # ------------------------------------------------------------- inspection

    def __iter__(self) -> Iterator[Schedule]:
        return iter(self.routes)

    def __len__(self) -> int:
        return len(self.routes)

    def __hash__(self) -> int:
        return hash((self.routes, self.unassigned))

    def route_for(self, vehicle: VehicleId) -> Schedule:
        """The schedule for one vehicle.

        Raises:
            KeyError: If the vehicle has no route in this solution. That is a
                programmer error -- every vehicle in the problem should have a
                route, empty or not.
        """
        try:
            return self._by_vehicle[vehicle]
        except KeyError:
            msg = f"No route for vehicle {vehicle!r} in this solution."
            raise KeyError(msg) from None

    def has_vehicle(self, vehicle: VehicleId) -> bool:
        """Whether this solution carries a route for the vehicle."""
        return vehicle in self._by_vehicle

    def vehicles(self) -> tuple[VehicleId, ...]:
        """Every vehicle with a route, in route order."""
        return tuple(schedule.vehicle for schedule in self.routes)

    def used_vehicles(self) -> tuple[VehicleId, ...]:
        """Vehicles whose route is non-empty."""
        return tuple(schedule.vehicle for schedule in self.routes if schedule)

    def assigned_requests(self) -> frozenset[RequestId]:
        """Every request placed on some route."""
        requests: set[RequestId] = set()
        for schedule in self.routes:
            requests |= schedule.requests()
        return frozenset(requests)

    def vehicle_of(self, request: RequestId) -> VehicleId | None:
        """Which vehicle serves a request, or ``None`` if it is unassigned."""
        for schedule in self.routes:
            if request in schedule.requests():
                return schedule.vehicle
        return None

    # --------------------------------------------------------- transformations

    def with_route(self, schedule: Schedule) -> Solution:
        """A copy in which one vehicle's route is replaced.

        Raises:
            KeyError: If the schedule's vehicle has no route here.
        """
        if schedule.vehicle not in self._by_vehicle:
            msg = f"No route for vehicle {schedule.vehicle!r} in this solution."
            raise KeyError(msg)
        routes = tuple(
            schedule if existing.vehicle == schedule.vehicle else existing
            for existing in self.routes
        )
        return Solution(routes, self.unassigned)

    def with_routes(self, schedules: Iterable[Schedule]) -> Solution:
        """A copy in which several vehicles' routes are replaced at once."""
        replacements = {schedule.vehicle: schedule for schedule in schedules}
        routes = tuple(replacements.get(existing.vehicle, existing) for existing in self.routes)
        return Solution(routes, self.unassigned)

    def with_unassigned(self, requests: Iterable[RequestId]) -> Solution:
        """A copy whose unassigned pool is exactly ``requests``."""
        return Solution(self.routes, frozenset(requests))

    def with_released(self, requests: Iterable[RequestId]) -> Solution:
        """A copy with ``requests`` removed from every route and added to the pool.

        This is the ruin half of ruin-and-recreate.
        """
        targets = frozenset(requests)
        if not targets:
            return self
        routes = tuple(schedule.without_requests(targets) for schedule in self.routes)
        return Solution(routes, self.unassigned | targets)

    def with_assigned(self, request: RequestId, schedule: Schedule) -> Solution:
        """A copy with a request taken out of the pool and a route replacing one vehicle's.

        The caller supplies the already-built schedule, so this does not need to
        know how the request was placed.
        """
        return self.with_route(schedule).with_unassigned(self.unassigned - {request})

    # ------------------------------------------------------------- convenience

    def editing(self) -> SolutionEditor:
        """A mutable editor over this solution, committed back at the end.

        Use for sequential construction, where rebinding once per request is
        noise::

            with solution.editing() as editor:
                for request in ordered:
                    editor.assign(vehicle, request, pickup, i, dropoff, j)
            solution = editor.commit()
        """
        return SolutionEditor(self)

    def describe(self) -> str:
        """A multi-line rendering of every route plus the unassigned pool."""
        lines = [schedule.describe() for schedule in self.routes]
        if self.unassigned:
            lines.append(f"unassigned: {' '.join(sorted(self.unassigned))}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return (
            f"Solution({len(self.used_vehicles())}/{len(self.routes)} vehicles used, "
            f"{len(self.unassigned)} unassigned)"
        )


class SolutionEditor:
    """A mutable editor over an immutable :class:`Solution`.

    Exists for the same reason as
    :class:`~fleetlab.domain.schedule.RouteBuffer`: sequential construction is
    clearer when it reads as a loop of mutations, and nothing outside this
    object can observe the intermediate states.
    """

    __slots__ = ("_order", "_routes", "_unassigned")

    def __init__(self, solution: Solution) -> None:
        self._order: list[VehicleId] = [schedule.vehicle for schedule in solution.routes]
        self._routes: dict[VehicleId, Schedule] = {
            schedule.vehicle: schedule for schedule in solution.routes
        }
        self._unassigned: set[RequestId] = set(solution.unassigned)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def route_for(self, vehicle: VehicleId) -> Schedule:
        """The current route for a vehicle."""
        return self._routes[vehicle]

    def set_route(self, schedule: Schedule) -> None:
        """Replace one vehicle's route."""
        if schedule.vehicle not in self._routes:
            msg = f"No route for vehicle {schedule.vehicle!r} in this solution."
            raise KeyError(msg)
        self._routes[schedule.vehicle] = schedule

    def assign(self, request: RequestId, schedule: Schedule) -> None:
        """Install a route that now contains ``request`` and drop it from the pool."""
        self.set_route(schedule)
        self._unassigned.discard(request)

    def release(self, request: RequestId) -> None:
        """Remove a request from whichever route holds it and return it to the pool."""
        for vehicle, schedule in self._routes.items():
            stripped = schedule.without_request(request)
            if stripped is not schedule:
                self._routes[vehicle] = stripped
        self._unassigned.add(request)

    def release_many(self, requests: Iterable[RequestId]) -> None:
        """Release several requests in one pass over the routes."""
        targets = frozenset(requests)
        if not targets:
            return
        for vehicle, schedule in self._routes.items():
            self._routes[vehicle] = schedule.without_requests(targets)
        self._unassigned |= targets

    @property
    def unassigned(self) -> frozenset[RequestId]:
        """The current unassigned pool."""
        return frozenset(self._unassigned)

    def commit(self) -> Solution:
        """Freeze the editor into an immutable solution."""
        return Solution(
            tuple(self._routes[vehicle] for vehicle in self._order),
            frozenset(self._unassigned),
        )
