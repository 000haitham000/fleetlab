"""Physical locations."""

from __future__ import annotations

from dataclasses import dataclass, field

from fleetlab.domain.ids import StopId


@dataclass(frozen=True, slots=True)
class Stop:
    """A location a vehicle can be at.

    Coordinates are optional: many benchmark instances supply a travel matrix
    and no geometry at all, and the framework must not require one to be
    invented.

    Attributes:
        id: Stable identifier, unique within an instance.
        x: First coordinate (longitude, easting, or abstract x).
        y: Second coordinate (latitude, northing, or abstract y).
        name: Human-readable label for logs and plots.
    """

    id: StopId
    x: float | None = None
    y: float | None = None
    name: str = field(default="", compare=False)

    @property
    def has_geometry(self) -> bool:
        """Whether both coordinates are present."""
        return self.x is not None and self.y is not None

    def coordinates(self) -> tuple[float, float]:
        """Return ``(x, y)``.

        Raises:
            ValueError: If the stop has no geometry. Callers that can work
                without coordinates should check :attr:`has_geometry` first.
        """
        if self.x is None or self.y is None:
            msg = f"Stop {self.id!r} has no coordinates."
            raise ValueError(msg)
        return (self.x, self.y)

    def __str__(self) -> str:
        return self.name or self.id
