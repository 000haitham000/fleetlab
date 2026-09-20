"""The :class:`Study`: one object an algorithm author holds.

The pure core is the right foundation, but nobody should have to assemble an
evaluation context, a constraint set, an objective and a cache by hand every
time they write a heuristic. A :class:`Study` bundles the four things that stay
fixed while an algorithm runs -- the instance, the service policy, the rules and
the objective -- and exposes the handful of questions an algorithm actually asks:

* what does this route look like in time?
* is it feasible, and if not by how much?
* what does this solution cost, and what is the cost made of?

It is also the natural unit of a comparison study. Two algorithms handed the
same :class:`Study` are provably solving the same problem under the same rules
and the same objective, which is the claim a results table needs to be able to
make.

``Study`` owns a memoising evaluator, so it is mutable in the boring sense that
its cache fills up. Nothing it holds is mutated, and evaluation remains pure --
the cache is an optimisation, not state. Use :meth:`Study.rebased` to get a
study with different vehicle start states, which is what the simulator does at
each decision epoch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fleetlab.domain.solution import Solution
from fleetlab.feasibility import standard_constraints
from fleetlab.objective import standard_objective
from fleetlab.timing.context import EvalContext
from fleetlab.timing.evaluator import Evaluator
from fleetlab.timing.policy import DEFAULT_POLICY
from fleetlab.timing.slack import ForwardSlack, forward_slack

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from fleetlab.domain.ids import RequestId, VehicleId
    from fleetlab.domain.problem import Problem
    from fleetlab.domain.schedule import Schedule
    from fleetlab.feasibility.base import ConstraintSet
    from fleetlab.feasibility.violation import Feasibility
    from fleetlab.objective.base import CostBreakdown, Objective
    from fleetlab.timing.context import VehicleStart
    from fleetlab.timing.policy import ServicePolicy
    from fleetlab.timing.timing import RouteTiming, SolutionTiming


class Study:
    """An instance, the rules applied to it, and what counts as better.

    Attributes are read-only; everything that looks like a mutation returns a
    new object.
    """

    __slots__ = ("_constraints", "_ctx", "_evaluator", "_objective", "_slack_cache")

    def __init__(
        self,
        problem: Problem,
        *,
        policy: ServicePolicy = DEFAULT_POLICY,
        constraints: ConstraintSet | None = None,
        objective: Objective | None = None,
        starts: Mapping[VehicleId, VehicleStart] | None = None,
    ) -> None:
        """Assemble a study.

        Args:
            problem: The instance.
            policy: How arrival turns into service. Defaults to the
                early-arrival convention.
            constraints: The rules to enforce. Defaults to
                :func:`~fleetlab.feasibility.standard_constraints`.
            objective: What to minimise. Defaults to
                :func:`~fleetlab.objective.standard_objective`.
            starts: Per-vehicle start states, for replanning mid-simulation.
        """
        self._ctx = EvalContext(problem=problem, policy=policy, starts=dict(starts or {}))
        self._constraints = constraints if constraints is not None else standard_constraints()
        self._objective = objective if objective is not None else standard_objective()
        self._evaluator = Evaluator(self._ctx)
        self._slack_cache: dict[Schedule, ForwardSlack] = {}

    # ------------------------------------------------------------- accessors

    @property
    def problem(self) -> Problem:
        """The instance being solved."""
        return self._ctx.problem

    @property
    def context(self) -> EvalContext:
        """The evaluation context every timing uses."""
        return self._ctx

    @property
    def constraints(self) -> ConstraintSet:
        """The rules this study enforces."""
        return self._constraints

    @property
    def objective(self) -> Objective:
        """What this study minimises."""
        return self._objective

    @property
    def evaluator(self) -> Evaluator:
        """The memoising evaluator."""
        return self._evaluator

    # -------------------------------------------------------------- evaluation

    def timing(self, schedule: Schedule) -> RouteTiming:
        """Timing for one route, from cache when available."""
        return self._evaluator.route(schedule)

    def timings(self, solution: Solution) -> SolutionTiming:
        """Timings for every route in a solution."""
        return self._evaluator.solution(solution)

    def slack(self, schedule: Schedule) -> ForwardSlack:
        """Forward time slack for one route, from cache when available.

        Safe to cache for the same reason the timing is: the route is immutable,
        so nothing can invalidate the array underneath it.
        """
        cached = self._slack_cache.get(schedule)
        if cached is not None:
            return cached
        computed = forward_slack(schedule, self.timing(schedule), self._ctx)
        self._slack_cache[schedule] = computed
        return computed

    # ------------------------------------------------------------ feasibility

    def check_route(self, schedule: Schedule) -> Feasibility:
        """Every way one route breaks the rules, with magnitudes."""
        return self._constraints.check_route(schedule, self.timing(schedule), self._ctx)

    def check(self, solution: Solution) -> Feasibility:
        """Every way a solution breaks the rules, with magnitudes."""
        return self._constraints.check_solution(solution, self.timings(solution), self._ctx)

    def is_route_feasible(self, schedule: Schedule) -> bool:
        """Whether one route satisfies every route rule. Stops at the first failure."""
        return self._constraints.is_route_feasible(schedule, self.timing(schedule), self._ctx)

    # ------------------------------------------------------------------- cost

    def route_cost(self, schedule: Schedule) -> float:
        """Weighted cost of one route. The unit of delta evaluation."""
        return self._objective.route_cost(schedule, self.timing(schedule), self._ctx)

    def cost(self, solution: Solution) -> float:
        """Weighted cost of a whole solution."""
        return self._objective.total(solution, self.timings(solution), self._ctx)

    def breakdown(self, solution: Solution) -> CostBreakdown:
        """Weighted cost plus the per-term figures behind it."""
        return self._objective.breakdown(solution, self.timings(solution), self._ctx)

    def penalised_cost(
        self,
        solution: Solution,
        weights: Mapping[str, float] | None = None,
        default: float = 1e6,
    ) -> float:
        """Cost plus weighted constraint violations.

        This is what lets a search move through infeasible space deliberately.
        Raising ``default`` over a run drives the search back to feasibility; a
        study that adapts the weights per constraint passes ``weights``.
        """
        return self.cost(solution) + self.check(solution).penalty(weights, default)

    # --------------------------------------------------------------- building

    def empty_solution(self, requests: Iterable[RequestId] | None = None) -> Solution:
        """A solution with every vehicle idle and every request unassigned."""
        pool = requests if requests is not None else self.problem.requests.keys()
        return Solution.empty(self.problem.vehicles.keys(), pool)

    def released_by(self, instant: float) -> frozenset[RequestId]:
        """Requests known to the operator at ``instant``.

        In a static study every request is released at zero. In a dynamic one
        this is what keeps an algorithm from planning around information it
        should not have.
        """
        return frozenset(
            request_id
            for request_id, request in self.problem.requests.items()
            if request.released_at <= instant
        )

    # ------------------------------------------------------------ derivation

    def rebased(self, starts: Mapping[VehicleId, VehicleStart]) -> Study:
        """A study identical to this one but with different vehicle start states.

        The simulator calls this at every decision epoch: the instance, rules and
        objective are unchanged, but the fleet is somewhere else now. A fresh
        study means a fresh cache, which is correct -- timings computed against
        the old positions no longer apply.
        """
        return Study(
            self.problem,
            policy=self._ctx.policy,
            constraints=self._constraints,
            objective=self._objective,
            starts=starts,
        )

    def with_objective(self, objective: Objective) -> Study:
        """A study with a different objective. Timings still apply, so the cache carries over."""
        derived = Study(
            self.problem,
            policy=self._ctx.policy,
            constraints=self._constraints,
            objective=objective,
            starts=self._ctx.starts,
        )
        derived._evaluator = self._evaluator
        derived._slack_cache = self._slack_cache
        return derived

    def with_constraints(self, constraints: ConstraintSet) -> Study:
        """A study with different rules. Timings still apply, so the cache carries over."""
        derived = Study(
            self.problem,
            policy=self._ctx.policy,
            constraints=constraints,
            objective=self._objective,
            starts=self._ctx.starts,
        )
        derived._evaluator = self._evaluator
        derived._slack_cache = self._slack_cache
        return derived

    def clear_caches(self) -> None:
        """Drop the timing and slack caches. Call between independent runs."""
        self._evaluator.clear()
        self._slack_cache.clear()

    def __str__(self) -> str:
        hits, misses = self._evaluator.statistics
        return (
            f"Study({self.problem.name!r}, {self._objective}, "
            f"{len(self._constraints.names())} rules, cache {hits}/{hits + misses})"
        )
