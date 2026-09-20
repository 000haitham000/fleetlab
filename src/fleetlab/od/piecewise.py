"""A time-dependent travel matrix that stays exactly representable in a MIP.

Travel time is constant within each of a finite set of time buckets -- the
morning peak, the inter-peak, the evening peak, and so on. That is enough to
capture the effect a time-dependent study cares about while remaining
linearisable: a mathematical program selects a bucket per arc with binary
variables and reads a constant travel time from it.

FIFO
----
A time-dependent matrix should normally satisfy the **first-in-first-out**
property: leaving later must never let you arrive earlier. Bucketed matrices can
violate it at bucket boundaries, which produces schedules that are optimal only
because the model found a loophole in the discretisation.
:meth:`PiecewiseODMatrix.fifo_violations` reports where that can happen, so a
study can check its data rather than discover it in the results.
"""

from __future__ import annotations

import bisect
import itertools
from typing import TYPE_CHECKING

from fleetlab.od.base import TimeDependence

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from fleetlab.domain.ids import StopId
    from fleetlab.domain.units import Distance, Duration, Instant


class PiecewiseODMatrix:
    """Travel times that are constant within each time bucket.

    Attributes:
        boundaries: Ascending bucket start instants. The first must be at or
            before the horizon start; each bucket runs until the next boundary,
            and the last runs to infinity.
    """

    __slots__ = ("_boundaries", "_default_distance", "_distances", "_durations")

    def __init__(
        self,
        boundaries: Sequence[Instant],
        durations: Mapping[tuple[StopId, StopId], Sequence[Duration]],
        distances: Mapping[tuple[StopId, StopId], Distance] | None = None,
        *,
        default_distance: Distance | None = None,
    ) -> None:
        """Build a bucketed matrix.

        Args:
            boundaries: Ascending bucket start instants.
            durations: Per stop pair, one travel time per bucket. The sequence
                length must equal the number of buckets.
            distances: Per stop pair path length. Time-independent by nature.
            default_distance: Returned for pairs absent from ``distances``.

        Raises:
            ValueError: If the boundaries are not strictly ascending, or a pair's
                duration sequence does not match the bucket count.
        """
        if not boundaries:
            msg = "A piecewise matrix needs at least one bucket boundary."
            raise ValueError(msg)
        if any(later <= earlier for earlier, later in itertools.pairwise(boundaries)):
            msg = f"Bucket boundaries must be strictly ascending, got {list(boundaries)!r}."
            raise ValueError(msg)

        self._boundaries = list(boundaries)
        bucket_count = len(self._boundaries)

        self._durations: dict[tuple[StopId, StopId], tuple[Duration, ...]] = {}
        for pair, profile in durations.items():
            if len(profile) != bucket_count:
                msg = (
                    f"Pair {pair!r} has {len(profile)} durations but there are "
                    f"{bucket_count} buckets."
                )
                raise ValueError(msg)
            self._durations[pair] = tuple(profile)

        self._distances = dict(distances) if distances is not None else {}
        self._default_distance = default_distance

    # ---------------------------------------------------------------- protocol

    @property
    def time_dependence(self) -> TimeDependence:
        """Always :attr:`~fleetlab.od.base.TimeDependence.PIECEWISE`."""
        return TimeDependence.PIECEWISE

    def duration(self, depart_at: Instant, origin: StopId, destination: StopId) -> Duration:
        """Travel time in whichever bucket ``depart_at`` falls into."""
        if origin == destination:
            return 0.0
        return self.bucket_duration(self.bucket_index(depart_at), origin, destination)

    def distance(self, origin: StopId, destination: StopId) -> Distance:
        """Path length between two stops."""
        if origin == destination:
            return 0.0
        value = self._distances.get((origin, destination), self._default_distance)
        if value is None:
            msg = f"No travel distance for ({origin!r} -> {destination!r})."
            raise KeyError(msg)
        return value

    # ----------------------------------------------------------- bucketed API

    def buckets(self) -> Sequence[tuple[Instant, Instant]]:
        """Half-open ``[start, end)`` intervals, the last ending at infinity."""
        from fleetlab.domain.units import HORIZON_INFINITY

        ends = [*self._boundaries[1:], HORIZON_INFINITY]
        return tuple(zip(self._boundaries, ends, strict=True))

    def bucket_index(self, depart_at: Instant) -> int:
        """Which bucket an instant falls into. Instants before the first bucket use it."""
        index = bisect.bisect_right(self._boundaries, depart_at) - 1
        return max(0, index)

    def bucket_duration(self, bucket: int, origin: StopId, destination: StopId) -> Duration:
        """Travel time within one bucket.

        Raises:
            KeyError: If the stop pair is unknown.
            IndexError: If the bucket index is out of range.
        """
        if origin == destination:
            return 0.0
        try:
            profile = self._durations[(origin, destination)]
        except KeyError:
            msg = f"No travel profile for ({origin!r} -> {destination!r})."
            raise KeyError(msg) from None
        return profile[bucket]

    # -------------------------------------------------------------- data check

    def fifo_violations(self) -> tuple[tuple[tuple[StopId, StopId, int], ...], str]:
        """Report boundaries where leaving later would let a vehicle arrive earlier.

        Returns:
            A tuple of offending ``(origin, destination, bucket)`` triples and a
            human-readable summary. An empty triple tuple means the matrix is
            FIFO-consistent at every boundary.
        """
        offenders: list[tuple[StopId, StopId, int]] = []
        bucket_windows = self.buckets()
        for (origin, destination), profile in self._durations.items():
            for index in range(len(profile) - 1):
                boundary = bucket_windows[index + 1][0]
                arrive_just_before = boundary + profile[index]
                arrive_at_boundary = boundary + profile[index + 1]
                if arrive_at_boundary < arrive_just_before:
                    offenders.append((origin, destination, index))
        summary = (
            "FIFO-consistent at every bucket boundary."
            if not offenders
            else f"{len(offenders)} boundary crossing(s) allow arriving earlier by leaving later."
        )
        return tuple(offenders), summary
