"""The origin-destination matrix protocol.

Travel time is the one input that decides whether a mathematical program can
represent the problem exactly, so the protocol is required to **declare its own
time dependence** rather than leaving callers to guess.

Why this matters
----------------
A time-dependent matrix -- ``duration(depart_at, a, b)`` -- makes travel time a
function of the departure instant. In the search lane that is fine: the
evaluator knows the departure instant because it walks the route forward. In a
mathematical program the departure instant *is a decision variable*, so travel
time becomes a function of a variable, and the model is no longer linear. Since
real traffic profiles are not convex either, it is not rescuable by convexity.

There are three honest positions, and :class:`TimeDependence` names them so the
MIP lane can act on the difference instead of silently linearising something it
should not:

``CONSTANT``
    Travel time is a fixed matrix. The mathematical program is exact.

``PIECEWISE``
    Travel time is constant within each of a finite set of time buckets. The
    mathematical program is exact *for that discretisation*, at the cost of
    bucket-selection binaries per arc.

``ARBITRARY``
    Travel time is an opaque callable. Simulation only. A mathematical program
    can still be built by freezing a profile, but the bound it produces is not
    valid for the true problem, and the framework refuses to pretend otherwise.

Distance is kept separate from duration. Conflating the two makes a
distance-minimising objective impossible to state once travel times vary with
traffic.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fleetlab.domain.ids import StopId
    from fleetlab.domain.units import Distance, Duration, Instant


class TimeDependence(enum.Enum):
    """How a matrix's travel times vary with departure time."""

    CONSTANT = "constant"
    PIECEWISE = "piecewise"
    ARBITRARY = "arbitrary"

    @property
    def is_linearisable(self) -> bool:
        """Whether a mathematical program over this matrix yields a valid bound."""
        return self is not TimeDependence.ARBITRARY


@runtime_checkable
class ODMatrix(Protocol):
    """Travel times and distances between stops."""

    @property
    def time_dependence(self) -> TimeDependence:
        """How this matrix's durations vary with departure time."""
        ...

    def duration(self, depart_at: Instant, origin: StopId, destination: StopId) -> Duration:
        """Travel time from ``origin`` to ``destination`` leaving at ``depart_at``.

        Implementations that are :attr:`TimeDependence.CONSTANT` must ignore
        ``depart_at`` entirely.
        """
        ...

    def distance(self, origin: StopId, destination: StopId) -> Distance:
        """Path length from ``origin`` to ``destination``, independent of time."""
        ...


@runtime_checkable
class BucketedODMatrix(ODMatrix, Protocol):
    """An OD matrix whose time dependence is piecewise constant over buckets.

    Implementing this is what lets the MIP lane build an exact model of a
    time-dependent instance.
    """

    def buckets(self) -> Sequence[tuple[Instant, Instant]]:
        """The half-open time buckets, in ascending order and covering the horizon."""
        ...

    def bucket_duration(self, bucket: int, origin: StopId, destination: StopId) -> Duration:
        """Travel time within one bucket."""
        ...


def mean_duration(
    matrix: ODMatrix,
    origin: StopId,
    destination: StopId,
    samples: Sequence[Instant],
) -> Duration:
    """Average a time-dependent duration over sample departure instants.

    Used when a study knowingly freezes a profile to obtain a (non-valid)
    mathematical-programming approximation of an ``ARBITRARY`` instance. It is a
    deliberate, named approximation rather than something the exporter does
    behind the caller's back.
    """
    if not samples:
        msg = "mean_duration needs at least one sample instant."
        raise ValueError(msg)
    total = sum(matrix.duration(at, origin, destination) for at in samples)
    return total / len(samples)
