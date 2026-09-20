"""Vehicle specifications.

A :class:`Vehicle` here is a *specification*, not a solution object. It holds
only what is true about the vehicle regardless of what it is asked to do: where
it starts and ends, when it is available, what it can carry, when its driver
rests.

It does not hold a schedule. A vehicle that owned its own route would also be
the constraint checker and the schedule evaluator -- three responsibilities that
change for different reasons and want different mutability. Here the route lives
in
:class:`~fleetlab.domain.schedule.Schedule`, evaluation in
:mod:`fleetlab.timing`, and feasibility in :mod:`fleetlab.feasibility`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fleetlab.domain.capacity import Capacity
from fleetlab.domain.ids import StopId, VehicleId
from fleetlab.domain.units import HORIZON_INFINITY, Duration, Instant, clock


@dataclass(frozen=True, slots=True)
class DriverBreak:
    """A period during which the vehicle is unavailable to serve actions.

    Breaks are honoured by the evaluator through
    :class:`~fleetlab.timing.policy.ServicePolicy`, and by the MIP lane as
    forbidden service intervals. A break that is stored but never enters a
    timing calculation is worse than no break at all: it reads as modelled.

    Attributes:
        start: When the break begins.
        end: When the break ends.
        stop: Where the break must be taken, if it is fixed to a location.
        flexible_by: How far the break may be shifted earlier or later to fit
            the schedule. Zero means it is pinned.
    """

    start: Instant
    end: Instant
    stop: StopId | None = None
    flexible_by: Duration = 0.0

    def __post_init__(self) -> None:
        if self.end < self.start:
            msg = f"Break ends before it starts: {self.start} .. {self.end}"
            raise ValueError(msg)

    @property
    def duration(self) -> Duration:
        """How long the break lasts."""
        return self.end - self.start

    def contains(self, instant: Instant) -> bool:
        """Whether the instant falls within the break, inclusive of both ends."""
        return self.start <= instant <= self.end

    def __str__(self) -> str:
        return f"break {clock(self.start)}-{clock(self.end)}"


@dataclass(frozen=True, slots=True)
class Vehicle:
    """What a vehicle is, independent of what it is doing.

    Attributes:
        id: Stable identifier, unique within an instance.
        start_stop: Where the vehicle begins its shift.
        end_stop: Where the vehicle must finish. The return leg is part of the
            evaluated schedule and counts toward the availability-end check.
        capacity: Per-dimension carrying limit.
        available_from: Start of the vehicle's shift.
        available_until: End of the vehicle's shift.
        breaks: Driver rest periods within the shift.
        fixed_cost: Cost incurred if this vehicle is used at all. Drives
            fleet-minimisation objectives.
        cost_per_minute: Marginal cost of shift duration.
        cost_per_distance: Marginal cost of distance travelled.
        max_shift_duration: Optional cap on elapsed time from leaving
            ``start_stop`` to arriving at ``end_stop``.
        skills: Capability tags. A study may register a constraint matching
            these against request attributes; the core does not interpret them.
    """

    id: VehicleId
    start_stop: StopId
    end_stop: StopId
    capacity: Capacity
    available_from: Instant = 0.0
    available_until: Instant = HORIZON_INFINITY
    breaks: tuple[DriverBreak, ...] = ()
    fixed_cost: float = 0.0
    cost_per_minute: float = 0.0
    cost_per_distance: float = 0.0
    max_shift_duration: Duration | None = None
    skills: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.available_until < self.available_from:
            msg = (
                f"Vehicle {self.id!r} availability ends before it starts: "
                f"{self.available_from} .. {self.available_until}"
            )
            raise ValueError(msg)

    @property
    def shift_window(self) -> tuple[Instant, Instant]:
        """The vehicle's availability as a pair."""
        return (self.available_from, self.available_until)

    def is_on_break(self, instant: Instant) -> bool:
        """Whether the vehicle is resting at the given instant."""
        return any(rest.contains(instant) for rest in self.breaks)

    def break_after(self, instant: Instant) -> DriverBreak | None:
        """The first break that begins at or after ``instant``, if any."""
        upcoming = [rest for rest in self.breaks if rest.start >= instant]
        return min(upcoming, key=lambda rest: rest.start) if upcoming else None

    def __str__(self) -> str:
        return f"vehicle {self.id}"
