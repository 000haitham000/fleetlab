"""A travel matrix that does not vary with time."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fleetlab.od.base import TimeDependence

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fleetlab.domain.ids import StopId
    from fleetlab.domain.units import Distance, Duration, Instant


class ConstantODMatrix:
    """Fixed travel times and distances, looked up by stop pair.

    This is the matrix a mathematical program can represent exactly, so it is
    the one to use for any study that wants a valid bound.
    """

    __slots__ = ("_default_distance", "_default_duration", "_distances", "_durations")

    def __init__(
        self,
        durations: Mapping[tuple[StopId, StopId], Duration],
        distances: Mapping[tuple[StopId, StopId], Distance] | None = None,
        *,
        default_duration: Duration | None = None,
        default_distance: Distance | None = None,
    ) -> None:
        """Build a constant matrix.

        Args:
            durations: Travel time per ``(origin, destination)`` pair.
            distances: Path length per pair. Defaults to the durations, which is
                only sensible when the study measures cost in time.
            default_duration: Returned for pairs absent from ``durations``. When
                ``None``, a missing pair raises, which is usually what you want
                while building an instance.
            default_distance: As above, for distances.
        """
        self._durations = dict(durations)
        self._distances = dict(distances) if distances is not None else dict(durations)
        self._default_duration = default_duration
        self._default_distance = default_distance

    @property
    def time_dependence(self) -> TimeDependence:
        """Always :attr:`~fleetlab.od.base.TimeDependence.CONSTANT`."""
        return TimeDependence.CONSTANT

    def duration(self, depart_at: Instant, origin: StopId, destination: StopId) -> Duration:  # noqa: ARG002
        """Travel time between two stops. ``depart_at`` is ignored by contract."""
        if origin == destination:
            return 0.0
        value = self._durations.get((origin, destination), self._default_duration)
        if value is None:
            msg = f"No travel duration for ({origin!r} -> {destination!r})."
            raise KeyError(msg)
        return value

    def distance(self, origin: StopId, destination: StopId) -> Distance:
        """Path length between two stops."""
        if origin == destination:
            return 0.0
        value = self._distances.get((origin, destination), self._default_distance)
        if value is None:
            msg = f"No travel distance for ({origin!r} -> {destination!r})."
            raise KeyError(msg)
        return value

    def stops(self) -> frozenset[StopId]:
        """Every stop mentioned by the matrix."""
        known: set[StopId] = set()
        for origin, destination in self._durations:
            known.add(origin)
            known.add(destination)
        return frozenset(known)
