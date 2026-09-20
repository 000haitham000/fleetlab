"""Violations: infeasibility reported as data, with a size.

Checking a schedule returns data, never an exception. Two reasons, and both are
load-bearing.

**You learn only yes or no.** An exception says the schedule is infeasible; it
never says by how much. Penalty-based metaheuristics -- guided local search, large
neighbourhood search with adaptive penalties, annealing that deliberately
crosses into infeasible territory -- all steer on the *margin*. Most competitive
pickup-and-delivery heuristics traverse infeasible space on purpose. Reporting a
boolean forecloses that entire family, which in a framework built to compare
algorithms is the most expensive thing on the list.

**The state has to be damaged to ask the question.** Throwing implies the
schedule was mutated first, then unwound. Here nothing is mutated and nothing is
unwound: checking is a pure read.

So a check returns :class:`Violation` objects carrying a magnitude in natural
units -- minutes late, seats over, kilograms over -- and the caller decides
whether to reject, repair or penalise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from fleetlab.domain.ids import RequestId, VehicleId


@dataclass(frozen=True, slots=True)
class Violation:
    """One way in which a schedule fails a constraint.

    Attributes:
        constraint: Name of the constraint that reported it.
        magnitude: How badly, in :attr:`unit`. Always positive -- a zero
            magnitude means no violation, and no object should be produced.
        unit: What the magnitude is measured in, for penalty weighting and for
            reports that mix time with capacity.
        vehicle: The route involved, if any.
        request: The request involved, if any.
        index: The schedule position involved, if any.
        detail: A short human-readable explanation for logs.
    """

    constraint: str
    magnitude: float
    unit: str = "minutes"
    vehicle: VehicleId | None = None
    request: RequestId | None = None
    index: int | None = None
    detail: str = ""

    def __str__(self) -> str:
        where = []
        if self.vehicle is not None:
            where.append(str(self.vehicle))
        if self.request is not None:
            where.append(str(self.request))
        if self.index is not None:
            where.append(f"#{self.index}")
        location = f" [{' '.join(where)}]" if where else ""
        detail = f" -- {self.detail}" if self.detail else ""
        return f"{self.constraint}: {self.magnitude:.4g} {self.unit} over{location}{detail}"


@dataclass(frozen=True, slots=True)
class Feasibility:
    """The result of checking a schedule or a solution.

    Attributes:
        violations: Everything that failed. Empty means feasible.
    """

    violations: tuple[Violation, ...] = ()

    @classmethod
    def ok(cls) -> Feasibility:
        """A result with no violations."""
        return cls(())

    @classmethod
    def of(cls, violations: Iterable[Violation]) -> Feasibility:
        """Collect an iterable of violations into a result."""
        return cls(tuple(violations))

    @property
    def feasible(self) -> bool:
        """Whether nothing was violated."""
        return not self.violations

    def __len__(self) -> int:
        return len(self.violations)

    def merged_with(self, other: Feasibility) -> Feasibility:
        """A result carrying both sets of violations."""
        return Feasibility(self.violations + other.violations)

    def penalty(self, weights: Mapping[str, float] | None = None, default: float = 1.0) -> float:
        """Weighted total of violation magnitudes.

        This is what a penalised objective adds to the true cost. Weights are
        keyed by constraint name, so a study can make time windows soft and
        capacity hard by weighting them decades apart -- and can adapt those
        weights during a run, which is how adaptive large neighbourhood search
        escapes infeasible basins.

        Args:
            weights: Per-constraint multipliers.
            default: Multiplier for constraints absent from ``weights``.
        """
        if weights is None:
            return sum(violation.magnitude for violation in self.violations)
        return sum(
            violation.magnitude * weights.get(violation.constraint, default)
            for violation in self.violations
        )

    def by_constraint(self) -> dict[str, tuple[Violation, ...]]:
        """Violations grouped by the constraint that produced them."""
        grouped: dict[str, list[Violation]] = {}
        for violation in self.violations:
            grouped.setdefault(violation.constraint, []).append(violation)
        return {name: tuple(items) for name, items in grouped.items()}

    def worst(self) -> Violation | None:
        """The single largest violation by magnitude, or ``None`` if feasible."""
        if not self.violations:
            return None
        return max(self.violations, key=lambda violation: violation.magnitude)

    def constraint_names(self) -> frozenset[str]:
        """Which constraints reported something."""
        return frozenset(violation.constraint for violation in self.violations)

    def describe(self) -> str:
        """A multi-line report, for logs and failing tests."""
        if not self.violations:
            return "feasible"
        lines = [f"{len(self.violations)} violation(s):"]
        lines.extend(f"  {violation}" for violation in self.violations)
        return "\n".join(lines)

    def __str__(self) -> str:
        if self.feasible:
            return "Feasibility(ok)"
        return f"Feasibility({len(self.violations)} violations, worst={self.worst()})"


FEASIBLE = Feasibility(())
"""Shared empty result, to avoid allocating one per feasible check."""
