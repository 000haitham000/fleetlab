"""Objectives: what a study is actually trying to minimise.

A framework whose purpose is to *compare* algorithms has to state explicitly
what "better" means. Constraints alone are not enough: without a shared and
explicit objective, two algorithms are not being compared -- two problems are.

Three properties the design needs, and why:

**Composable.** Real objectives are weighted sums: distance, plus lateness, plus
passenger inconvenience, plus fleet size. Each term is its own object, so a
study states its trade-offs as data rather than as an edited function.

**Decomposable.** Total cost is the sum of per-route costs plus a fleet-level
remainder. That split is what makes delta evaluation possible: a move touching
two routes needs only those two re-costed, not the whole fleet.

**Reportable.** :class:`CostBreakdown` keeps the per-term figures, so a results
table can show *why* one algorithm beat another rather than only that it did.

Terms that can also be stated linearly implement
:class:`LinearObjectiveTerm`, and the mathematical-programming lane assembles
its objective from exactly the same term list -- so the two lanes are provably
optimising the same function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from fleetlab.linear import LinExpr

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from fleetlab.domain.schedule import Schedule
    from fleetlab.domain.solution import Solution
    from fleetlab.mathprog.variables import PickupDeliveryVars
    from fleetlab.timing.context import EvalContext
    from fleetlab.timing.timing import RouteTiming, SolutionTiming


@runtime_checkable
class ObjectiveTerm(Protocol):
    """One component of an objective function."""

    @property
    def name(self) -> str:
        """Short identifier, used as the key in a cost breakdown."""
        ...

    def route_cost(
        self,
        schedule: Schedule,
        timing: RouteTiming,
        ctx: EvalContext,
    ) -> float:
        """Cost attributable to one route. Return ``0.0`` for fleet-level terms."""
        ...

    def fleet_cost(
        self,
        solution: Solution,
        timings: SolutionTiming,
        ctx: EvalContext,
    ) -> float:
        """Cost not attributable to any single route. Return ``0.0`` for route terms."""
        ...


@runtime_checkable
class LinearObjectiveTerm(Protocol):
    """An objective term that can also be written as a linear expression."""

    def to_expr(
        self,
        variables: PickupDeliveryVars,
        ctx: EvalContext,
    ) -> LinExpr:
        """The term as a linear expression over the model's variables."""
        ...


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """A total alongside the per-term figures that produced it.

    Attributes:
        total: The weighted sum.
        terms: Weighted contribution per term name.
        raw: Unweighted contribution per term name. Keeping both means a results
            table can report "12 minutes of lateness, weighted 100" rather than
            an opaque 1200.
    """

    total: float
    terms: Mapping[str, float]
    raw: Mapping[str, float]

    def __float__(self) -> float:
        return self.total

    def describe(self) -> str:
        """A multi-line report of where the cost came from."""
        lines = [f"total {self.total:.4f}"]
        for name in sorted(self.terms):
            lines.append(f"  {name:<22} {self.terms[name]:>12.4f}  (raw {self.raw[name]:.4f})")
        return "\n".join(lines)

    def __str__(self) -> str:
        return f"CostBreakdown({self.total:.4f})"


@dataclass(frozen=True, slots=True)
class Objective:
    """A weighted sum of objective terms.

    Attributes:
        terms: The components.
        weights: Multiplier per term name. Terms absent from the mapping are
            weighted ``1.0``.
    """

    terms: tuple[ObjectiveTerm, ...]
    weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = [term.name for term in self.terms]
        if len(set(names)) != len(names):
            msg = f"Objective has duplicate term names: {names!r}"
            raise ValueError(msg)

    def weight_of(self, name: str) -> float:
        """The multiplier for one term."""
        return self.weights.get(name, 1.0)

    # ------------------------------------------------------------- evaluation

    def route_cost(
        self,
        schedule: Schedule,
        timing: RouteTiming,
        ctx: EvalContext,
    ) -> float:
        """Weighted cost of one route. This is what delta evaluation calls."""
        return sum(
            self.weight_of(term.name) * term.route_cost(schedule, timing, ctx)
            for term in self.terms
        )

    def total(
        self,
        solution: Solution,
        timings: SolutionTiming,
        ctx: EvalContext,
    ) -> float:
        """Weighted cost of a whole solution."""
        running = 0.0
        for term in self.terms:
            weight = self.weight_of(term.name)
            for schedule in solution.routes:
                running += weight * term.route_cost(schedule, timings[schedule.vehicle], ctx)
            running += weight * term.fleet_cost(solution, timings, ctx)
        return running

    def breakdown(
        self,
        solution: Solution,
        timings: SolutionTiming,
        ctx: EvalContext,
    ) -> CostBreakdown:
        """Weighted total plus the per-term figures behind it."""
        weighted: dict[str, float] = {}
        raw: dict[str, float] = {}
        for term in self.terms:
            amount = sum(
                term.route_cost(schedule, timings[schedule.vehicle], ctx)
                for schedule in solution.routes
            )
            amount += term.fleet_cost(solution, timings, ctx)
            raw[term.name] = amount
            weighted[term.name] = self.weight_of(term.name) * amount
        return CostBreakdown(sum(weighted.values()), weighted, raw)

    # ---------------------------------------------- mathematical-programming lane

    def linear_terms(self) -> tuple[tuple[str, LinearObjectiveTerm], ...]:
        """Terms that can be written linearly, with their names."""
        return tuple(
            (term.name, term) for term in self.terms if isinstance(term, LinearObjectiveTerm)
        )

    def non_linear_terms(self) -> tuple[str, ...]:
        """Names of terms that cannot be written linearly.

        A mathematical-programming run that ignores one of these is optimising a
        different function from the search lane, so the exporter reports them
        rather than dropping them quietly.
        """
        return tuple(term.name for term in self.terms if not isinstance(term, LinearObjectiveTerm))

    def to_expr(
        self,
        variables: PickupDeliveryVars,
        ctx: EvalContext,
    ) -> LinExpr:
        """The objective as a linear expression, from the linearisable terms."""
        return LinExpr.sum(
            term.to_expr(variables, ctx) * self.weight_of(name)
            for name, term in self.linear_terms()
        )

    # ------------------------------------------------------------ composition

    def reweighted(self, **weights: float) -> Objective:
        """A copy with some weights changed.

        Adaptive penalty schemes call this every few iterations, so it is a
        cheap copy rather than a mutation.
        """
        merged = dict(self.weights)
        merged.update(weights)
        return Objective(self.terms, merged)

    def with_terms(self, terms: Iterable[ObjectiveTerm]) -> Objective:
        """A copy carrying further terms."""
        return Objective((*self.terms, *terms), self.weights)

    def names(self) -> Sequence[str]:
        """Every term name, in order."""
        return [term.name for term in self.terms]

    def __str__(self) -> str:
        parts = [f"{self.weight_of(name)}*{name}" for name in self.names()]
        return f"Objective({' + '.join(parts) or 'empty'})"
