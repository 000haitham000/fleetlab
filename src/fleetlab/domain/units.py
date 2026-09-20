"""Time units for the whole framework.

Every instant and duration in ``fleetlab`` is a plain ``float`` holding **minutes
since the instance epoch**. This is deliberate and it is load-bearing:

* A mathematical program needs numbers. ``datetime`` arithmetic cannot appear in
  a constraint row, so any design that carries wall-clock objects through the
  core has to convert at the MIP boundary anyway -- and conversion at a boundary
  is exactly where the two lanes drift apart.
* Float arithmetic is roughly two orders of magnitude cheaper than timezone-aware
  datetime arithmetic, and the evaluator runs in the innermost loop of every
  search.

Wall-clock values live only at the IO boundary (:mod:`fleetlab.io.instance`),
where :class:`Epoch` converts them in and out.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import TypeAlias

#: An absolute point in time, as minutes since the instance epoch.
Instant: TypeAlias = float

#: An elapsed span of time, in minutes. Always non-negative in valid data.
Duration: TypeAlias = float

#: A spatial distance, in whatever unit the instance's OD matrix uses (km by convention).
Distance: TypeAlias = float

MINUTE: Duration = 1.0
HOUR: Duration = 60.0
DAY: Duration = 1440.0

#: Sentinel for "no upper bound". Large enough to dominate any real horizon,
#: small enough that arithmetic on it stays exact in float64 and it can be used
#: as a big-M without wrecking a solver's numerics.
HORIZON_INFINITY: Instant = 1.0e7


@dataclass(frozen=True, slots=True)
class Epoch:
    """Converts between wall-clock datetimes and instance-relative minutes.

    An instance fixes one epoch. Everything inside the framework is measured
    from it, so that the core never touches a timezone.

    Attributes:
        origin: The wall-clock datetime that corresponds to instant ``0.0``.
    """

    origin: _dt.datetime

    def to_instant(self, when: _dt.datetime) -> Instant:
        """Convert a wall-clock datetime to minutes since this epoch."""
        return (when - self.origin).total_seconds() / 60.0

    def to_datetime(self, instant: Instant) -> _dt.datetime:
        """Convert minutes since this epoch back to a wall-clock datetime."""
        return self.origin + _dt.timedelta(minutes=instant)

    def format(self, instant: Instant) -> str:
        """Render an instant as ``HH:MM:SS`` for logs and reprs."""
        return self.to_datetime(instant).strftime("%H:%M:%S")


def clock(instant: Instant) -> str:
    """Render an instant as ``HH:MM`` relative to midnight of day zero.

    Used only for human-readable output where no :class:`Epoch` is at hand.
    """
    total = round(instant)
    sign = "-" if total < 0 else ""
    total = abs(total)
    return f"{sign}{total // 60:02d}:{total % 60:02d}"
