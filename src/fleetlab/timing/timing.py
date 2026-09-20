"""Timing results.

One forward pass over a route produces *everything* time-related about it at
once: arrivals, service starts, departures, driver waiting, onboard times,
lateness, the load profile and the return to the end depot.

Producing them together matters for two reasons. Computing each separately
means walking the route once per quantity, so asking for all of a route's
arrival times costs O(n^2). And a quantity that is not represented has to be
reconstructed by subtracting two that are -- onboard time above all, which is a
constraint in dial-a-ride work and an objective term in most studies. It earns
a field.

These objects are immutable and cheap to keep, which is what allows a route's
timing to be cached and its forward slack to stay valid (see
:mod:`fleetlab.timing.slack`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fleetlab.domain.units import Distance, Duration, Instant, clock

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fleetlab.domain.capacity import Capacity
    from fleetlab.domain.ids import RequestId, VehicleId


@dataclass(frozen=True, slots=True)
class StopTiming:
    """When a vehicle arrives, starts service and departs at one action.

    Attributes:
        index: Position in the schedule this timing belongs to.
        arrival: When the vehicle reaches the stop.
        service_start: When service actually begins. Later than ``arrival`` when
            the vehicle is early and the policy holds it, or when a driver break
            intervenes.
        departure: When the vehicle leaves, that is ``service_start`` plus dwell.
    """

    index: int
    arrival: Instant
    service_start: Instant
    departure: Instant

    @property
    def wait(self) -> Duration:
        """Idle time between arriving and starting service."""
        return self.service_start - self.arrival

    @property
    def dwell(self) -> Duration:
        """Time spent in service."""
        return self.departure - self.service_start

    def __str__(self) -> str:
        return (
            f"#{self.index} arr {clock(self.arrival)} "
            f"svc {clock(self.service_start)} dep {clock(self.departure)}"
        )


@dataclass(frozen=True, slots=True)
class RouteTiming:
    """The complete temporal picture of one route.

    Attributes:
        vehicle: Whose route this is.
        depot_departure: When the vehicle leaves its start stop.
        stops: One entry per action, in schedule order.
        loads: Load carried *after* completing each action, in schedule order.
        return_arrival: When the vehicle reaches its end stop. The return leg
            is modelled, so a schedule that cannot get the vehicle home in time
            is detectable.
        travel_distance: Total distance including the return leg.
        travel_time: Total time spent moving, excluding waiting and dwell.
        request_positions: Pickup and dropoff index per request on this route.
        direct_durations: Origin-to-destination travel time per request, taken
            at that request's actual pickup departure. The reference for excess
            onboard time.
    """

    vehicle: VehicleId
    depot_departure: Instant
    stops: tuple[StopTiming, ...]
    loads: tuple[Capacity, ...]
    return_arrival: Instant
    travel_distance: Distance
    travel_time: Duration
    request_positions: Mapping[RequestId, tuple[int, int]] = field(default_factory=dict)
    direct_durations: Mapping[RequestId, Duration] = field(default_factory=dict)

    # ------------------------------------------------------------ per-position

    def __len__(self) -> int:
        return len(self.stops)

    def __getitem__(self, index: int) -> StopTiming:
        return self.stops[index]

    @property
    def is_empty(self) -> bool:
        """Whether the route serves nothing."""
        return not self.stops

    def arrival(self, index: int) -> Instant:
        """Arrival instant at one position."""
        return self.stops[index].arrival

    def service_start(self, index: int) -> Instant:
        """Service start instant at one position."""
        return self.stops[index].service_start

    def departure(self, index: int) -> Instant:
        """Departure instant at one position."""
        return self.stops[index].departure

    def wait(self, index: int) -> Duration:
        """Driver idle time at one position."""
        return self.stops[index].wait

    # ------------------------------------------------------------- per-request

    def onboard_time(self, request: RequestId) -> Duration | None:
        """Time between leaving the origin and beginning service at the destination.

        Returns ``None`` when the request is not fully present on this route. A
        half-placed request is a legitimate intermediate state in the search
        lane, not an error.
        """
        positions = self.request_positions.get(request)
        if positions is None:
            return None
        pickup, dropoff = positions
        return self.stops[dropoff].service_start - self.stops[pickup].departure

    def excess_onboard_time(self, request: RequestId) -> Duration | None:
        """Onboard time above the direct origin-to-destination travel time.

        This is the detour a shared ride imposes, and the usual form of the
        quality-of-service term in a dial-a-ride objective.
        """
        onboard = self.onboard_time(request)
        if onboard is None:
            return None
        direct = self.direct_durations.get(request, 0.0)
        return max(0.0, onboard - direct)

    # ------------------------------------------------------------- aggregates

    @property
    def route_duration(self) -> Duration:
        """Elapsed time from leaving the start stop to reaching the end stop."""
        return self.return_arrival - self.depot_departure

    @property
    def total_wait(self) -> Duration:
        """Total driver idle time across the route."""
        return sum(stop.wait for stop in self.stops)

    @property
    def total_dwell(self) -> Duration:
        """Total time in service across the route."""
        return sum(stop.dwell for stop in self.stops)

    def total_onboard_time(self) -> Duration:
        """Summed onboard time over every request fully served by this route."""
        total = 0.0
        for request in self.request_positions:
            onboard = self.onboard_time(request)
            if onboard is not None:
                total += onboard
        return total

    def total_excess_onboard_time(self) -> Duration:
        """Summed excess onboard time over every request fully served by this route."""
        total = 0.0
        for request in self.request_positions:
            excess = self.excess_onboard_time(request)
            if excess is not None:
                total += excess
        return total

    def peak_load(self) -> Capacity | None:
        """The largest load carried, dimension by dimension, or ``None`` if empty."""
        if not self.loads:
            return None
        peak = self.loads[0]
        for load in self.loads[1:]:
            peak = type(peak)(
                tuple(max(a, b) for a, b in zip(peak.values, load.values, strict=True))
            )
        return peak

    def describe(self) -> str:
        """A multi-line table of the route's timings, for logs and debugging."""
        lines = [
            f"{self.vehicle}: depart {clock(self.depot_departure)}, "
            f"return {clock(self.return_arrival)}, "
            f"drive {self.travel_time:.1f}m, wait {self.total_wait:.1f}m"
        ]
        lines.extend(f"  {stop}" for stop in self.stops)
        return "\n".join(lines)

    def __str__(self) -> str:
        return f"RouteTiming({self.vehicle}: {len(self.stops)} stops, {self.route_duration:.1f}m)"


@dataclass(frozen=True, slots=True)
class SolutionTiming:
    """Timings for every route in a solution.

    Attributes:
        routes: One :class:`RouteTiming` per vehicle.
    """

    routes: Mapping[VehicleId, RouteTiming]

    def __getitem__(self, vehicle: VehicleId) -> RouteTiming:
        return self.routes[vehicle]

    def __iter__(self) -> object:
        return iter(self.routes.values())

    def values(self) -> tuple[RouteTiming, ...]:
        """Every route timing."""
        return tuple(self.routes.values())

    @property
    def total_travel_time(self) -> Duration:
        """Summed moving time across the fleet."""
        return sum(route.travel_time for route in self.routes.values())

    @property
    def total_travel_distance(self) -> Distance:
        """Summed distance across the fleet."""
        return sum(route.travel_distance for route in self.routes.values())

    @property
    def total_wait(self) -> Duration:
        """Summed driver idle time across the fleet."""
        return sum(route.total_wait for route in self.routes.values())
