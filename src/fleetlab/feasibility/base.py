"""Constraint protocols: one object, two renderings.

This is the change that lets a single framework serve both a metaheuristic and a
mathematical program without stating the rules twice.

A constraint knows its own semantics. It can express them in two ways:

``check_route`` / ``check_solution``
    The sequence is **known**, so evaluate it and report what is wrong and by
    how much.

``to_model``
    The sequence is **unknown** -- it is what the solver is choosing -- so emit
    rows that constrain it.

The second signature is the one that is easy to miss. A mathematical program
never asks "given this order, when do we arrive?" -- the order is its *output*,
not its input. A design that can only answer that question therefore has
nothing to offer a solver, however well it answers it.

Keeping both faces on one object is also the only way the two lanes stay
honest. Stating a time window in a checker and again in a formulation is two
places to get it wrong, and the study that compares them would be comparing two
different problems.

A constraint that cannot be linearised simply does not implement
:class:`Linearisable`. :meth:`ConstraintSet.non_linearisable` names them, so the
mathematical-programming lane can refuse loudly rather than quietly solving a
different problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from fleetlab.feasibility.violation import Feasibility

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from fleetlab.domain.schedule import Schedule
    from fleetlab.domain.solution import Solution
    from fleetlab.linear import Model
    from fleetlab.mathprog.variables import PickupDeliveryVars
    from fleetlab.timing.context import EvalContext
    from fleetlab.timing.timing import RouteTiming, SolutionTiming

    from .violation import Violation


@runtime_checkable
class RouteConstraint(Protocol):
    """A rule that can be judged by looking at one route in isolation."""

    @property
    def name(self) -> str:
        """Short identifier, used in violation reports and penalty weights."""
        ...

    def check_route(
        self,
        schedule: Schedule,
        timing: RouteTiming,
        ctx: EvalContext,
    ) -> Iterator[Violation]:
        """Yield a violation for each way this route breaks the rule.

        Must not raise for a modelling reason and must not mutate anything.
        Yielding nothing means the route satisfies the rule.
        """
        ...


@runtime_checkable
class SolutionConstraint(Protocol):
    """A rule that needs the whole fleet to judge."""

    @property
    def name(self) -> str:
        """Short identifier, used in violation reports and penalty weights."""
        ...

    def check_solution(
        self,
        solution: Solution,
        timings: SolutionTiming,
        ctx: EvalContext,
    ) -> Iterator[Violation]:
        """Yield a violation for each way this solution breaks the rule."""
        ...


@runtime_checkable
class Linearisable(Protocol):
    """A rule that can also be stated as rows of a mathematical program."""

    def to_model(
        self,
        model: Model,
        variables: PickupDeliveryVars,
        ctx: EvalContext,
    ) -> None:
        """Append the rows expressing this rule to ``model``."""
        ...


@dataclass(frozen=True, slots=True)
class ConstraintSet:
    """The rules one study has chosen to enforce.

    Which constraints are active is a study's decision, not the framework's. A
    dial-a-ride study registers
    :class:`~fleetlab.feasibility.onboard.MaxOnboardTime`; a parcel study
    usually does not, and may register handling rules of its own. Making the set
    explicit is what keeps the framework general across both without
    ``if moving_people`` anywhere in it.

    Attributes:
        route_rules: Constraints judged per route.
        solution_rules: Constraints judged across the fleet.
    """

    route_rules: tuple[RouteConstraint, ...] = ()
    solution_rules: tuple[SolutionConstraint, ...] = ()

    # ------------------------------------------------------------ search lane

    def check_route(
        self,
        schedule: Schedule,
        timing: RouteTiming,
        ctx: EvalContext,
    ) -> Feasibility:
        """Check one route against every route rule."""
        violations: list[Violation] = []
        for rule in self.route_rules:
            violations.extend(rule.check_route(schedule, timing, ctx))
        return Feasibility(tuple(violations))

    def check_solution(
        self,
        solution: Solution,
        timings: SolutionTiming,
        ctx: EvalContext,
    ) -> Feasibility:
        """Check every route, then the fleet-level rules."""
        violations: list[Violation] = []
        for schedule in solution.routes:
            timing = timings[schedule.vehicle]
            for rule in self.route_rules:
                violations.extend(rule.check_route(schedule, timing, ctx))
        for fleet_rule in self.solution_rules:
            violations.extend(fleet_rule.check_solution(solution, timings, ctx))
        return Feasibility(tuple(violations))

    def first_violation(
        self,
        schedule: Schedule,
        timing: RouteTiming,
        ctx: EvalContext,
    ) -> Violation | None:
        """Stop at the first violation found.

        Faster than :meth:`check_route` when the caller only needs a verdict --
        an acceptance test that rejects anything infeasible, for instance. Use
        :meth:`check_route` whenever the magnitudes matter.
        """
        for rule in self.route_rules:
            for violation in rule.check_route(schedule, timing, ctx):
                return violation
        return None

    def is_route_feasible(
        self,
        schedule: Schedule,
        timing: RouteTiming,
        ctx: EvalContext,
    ) -> bool:
        """Whether a route satisfies every route rule."""
        return self.first_violation(schedule, timing, ctx) is None

    # ------------------------------------------------- mathematical-programming lane

    def linearisable(self) -> tuple[Linearisable, ...]:
        """Every registered rule that can be rendered into a model."""
        rules: list[Linearisable] = []
        for rule in (*self.route_rules, *self.solution_rules):
            if isinstance(rule, Linearisable):
                rules.append(rule)
        return tuple(rules)

    def non_linearisable(self) -> tuple[str, ...]:
        """Names of rules that cannot be rendered into a model.

        A mathematical-programming run must either drop these deliberately or
        refuse. Silently omitting them would produce a bound for a different,
        looser problem -- which is exactly the kind of result that survives into
        a paper unnoticed.
        """
        names: list[str] = []
        for route_rule in self.route_rules:
            if not isinstance(route_rule, Linearisable):
                names.append(route_rule.name)
        for fleet_rule in self.solution_rules:
            if not isinstance(fleet_rule, Linearisable):
                names.append(fleet_rule.name)
        return tuple(names)

    def apply_to_model(
        self,
        model: Model,
        variables: PickupDeliveryVars,
        ctx: EvalContext,
    ) -> None:
        """Render every linearisable rule into ``model``."""
        for rule in self.linearisable():
            rule.to_model(model, variables, ctx)

    # ------------------------------------------------------------ composition

    def with_rules(
        self,
        route_rules: Iterable[RouteConstraint] = (),
        solution_rules: Iterable[SolutionConstraint] = (),
    ) -> ConstraintSet:
        """A copy with further rules registered."""
        return ConstraintSet(
            route_rules=(*self.route_rules, *route_rules),
            solution_rules=(*self.solution_rules, *solution_rules),
        )

    def without(self, *names: str) -> ConstraintSet:
        """A copy with the named rules removed."""
        drop = frozenset(names)
        return ConstraintSet(
            route_rules=tuple(rule for rule in self.route_rules if rule.name not in drop),
            solution_rules=tuple(rule for rule in self.solution_rules if rule.name not in drop),
        )

    def names(self) -> tuple[str, ...]:
        """Every registered rule's name."""
        return tuple(
            [rule.name for rule in self.route_rules] + [rule.name for rule in self.solution_rules]
        )

    def __str__(self) -> str:
        return f"ConstraintSet({', '.join(self.names()) or 'empty'})"


__all__ = [
    "ConstraintSet",
    "Feasibility",
    "Linearisable",
    "RouteConstraint",
    "SolutionConstraint",
]
