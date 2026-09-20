"""Time windows.

A window is stored as an **absolute** ``[earliest, latest]`` pair, not as a pair
of tolerances around a requested time.

Storing tolerances instead, and resolving them wherever they are used, invites
a specific and hard-to-see bug: the offsets get applied against different base
times in different places -- the requested time here, the promised time there --
so the same window quietly means different things depending on which code
asked. Resolving once, at construction, removes that whole class of bug. It is
also exactly the ``e_i`` and ``l_i`` a mathematical program needs.

:func:`window_around` is provided so instances that are *authored* as
offsets-around-a-requested-time can still be written that way; the resolution
happens once, at construction.
"""

from __future__ import annotations

from dataclasses import dataclass

from fleetlab.domain.units import HORIZON_INFINITY, Duration, Instant, clock


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """An absolute closed interval during which service may begin.

    Attributes:
        earliest: Service may not begin before this instant.
        latest: Service may not begin after this instant.
    """

    earliest: Instant
    latest: Instant

    def __post_init__(self) -> None:
        if self.latest < self.earliest:
            msg = f"Time window is empty: earliest={self.earliest} > latest={self.latest}"
            raise ValueError(msg)

    @property
    def width(self) -> Duration:
        """How wide the window is, in minutes."""
        return self.latest - self.earliest

    def contains(self, instant: Instant) -> bool:
        """Whether ``instant`` falls inside the closed interval."""
        return self.earliest <= instant <= self.latest

    def earliness(self, instant: Instant) -> Duration:
        """How far before :attr:`earliest` the instant falls; zero if not early."""
        return max(0.0, self.earliest - instant)

    def lateness(self, instant: Instant) -> Duration:
        """How far past :attr:`latest` the instant falls; zero if not late.

        This is the magnitude a time-window violation reports, and what a
        penalty-based metaheuristic steers on.
        """
        return max(0.0, instant - self.latest)

    def intersect(self, other: TimeWindow) -> TimeWindow | None:
        """The overlap of two windows, or ``None`` if they do not overlap.

        Returns ``None`` rather than raising: an empty intersection is a
        modelling outcome (these two things cannot both happen), not a bug.
        """
        earliest = max(self.earliest, other.earliest)
        latest = min(self.latest, other.latest)
        if latest < earliest:
            return None
        return TimeWindow(earliest, latest)

    def shifted(self, by: Duration) -> TimeWindow:
        """The same window moved ``by`` minutes later."""
        return TimeWindow(self.earliest + by, self.latest + by)

    def __str__(self) -> str:
        return f"[{clock(self.earliest)}, {clock(self.latest)}]"


def window_around(
    target: Instant,
    *,
    tolerance_early: Duration = 0.0,
    tolerance_late: Duration = 0.0,
) -> TimeWindow:
    """Resolve a target time plus tolerances into an absolute window.

    Args:
        target: The requested or promised instant.
        tolerance_early: How far before ``target`` service may begin.
        tolerance_late: How far after ``target`` service may begin.
    """
    return TimeWindow(target - tolerance_early, target + tolerance_late)


UNBOUNDED = TimeWindow(0.0, HORIZON_INFINITY)
"""A window that constrains nothing. Use for actions with no timing requirement."""
