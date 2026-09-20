"""A matrix derived from stop geometry.

Useful for generated instances, tests and demos, where authoring a full matrix
would be noise. Real studies should use a measured matrix.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fleetlab.od.base import TimeDependence

if TYPE_CHECKING:
    from collections.abc import Iterable

    from fleetlab.domain.ids import StopId
    from fleetlab.domain.stop import Stop
    from fleetlab.domain.units import Distance, Duration, Instant


class EuclideanODMatrix:
    """Straight-line distances at a fixed speed.

    Attributes are cached on construction so repeated lookups do not recompute
    square roots in the evaluator's inner loop.
    """

    __slots__ = ("_coordinates", "_distance_cache", "_speed")

    def __init__(self, stops: Iterable[Stop], speed: float = 1.0) -> None:
        """Build a Euclidean matrix.

        Args:
            stops: Stops with coordinates. Stops without geometry are rejected.
            speed: Distance units travelled per minute.

        Raises:
            ValueError: If ``speed`` is not positive, or a stop lacks coordinates.
        """
        if speed <= 0:
            msg = f"Speed must be positive, got {speed}."
            raise ValueError(msg)
        self._speed = speed
        self._coordinates: dict[StopId, tuple[float, float]] = {}
        for stop in stops:
            if not stop.has_geometry:
                msg = f"Stop {stop.id!r} has no coordinates; cannot build a Euclidean matrix."
                raise ValueError(msg)
            self._coordinates[stop.id] = stop.coordinates()
        self._distance_cache: dict[tuple[StopId, StopId], float] = {}

    @property
    def speed(self) -> float:
        """Distance units travelled per minute."""
        return self._speed

    @property
    def time_dependence(self) -> TimeDependence:
        """Always :attr:`~fleetlab.od.base.TimeDependence.CONSTANT`."""
        return TimeDependence.CONSTANT

    def duration(self, depart_at: Instant, origin: StopId, destination: StopId) -> Duration:  # noqa: ARG002
        """Travel time at the fixed speed. ``depart_at`` is ignored by contract."""
        return self.distance(origin, destination) / self._speed

    def distance(self, origin: StopId, destination: StopId) -> Distance:
        """Straight-line distance between two stops."""
        if origin == destination:
            return 0.0
        key = (origin, destination)
        cached = self._distance_cache.get(key)
        if cached is not None:
            return cached
        try:
            (x1, y1) = self._coordinates[origin]
            (x2, y2) = self._coordinates[destination]
        except KeyError as error:
            msg = f"Stop {error.args[0]!r} is not in this matrix."
            raise KeyError(msg) from None
        value = math.hypot(x2 - x1, y2 - y1)
        self._distance_cache[key] = value
        self._distance_cache[(destination, origin)] = value
        return value
