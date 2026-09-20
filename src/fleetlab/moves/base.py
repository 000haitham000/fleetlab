"""Moves: neighbourhood steps as data.

A move is an object, not a method call. It says what it would do and which
routes it touches; applying it returns a new solution and leaves the old one
intact.

Three things that buys, none of which is available when a move is a mutation:

* **Delta evaluation.** A move declares its touched vehicles, so only those
  routes need re-costing. That is the difference between O(routes) and O(1) per
  candidate in an inner loop.
* **Rejection is free.** There is nothing to undo, so no inverse operation has
  to be written and tested per move type -- which is where mutable neighbourhood
  code usually grows its subtlest bugs.
* **Moves are inspectable.** They can be logged, replayed, counted by type for
  an adaptive operator selector, and shipped to a worker process.

Note what is *not* here: nothing moves a single action between routes. Both ends
of a request travel together, always. That invariant lives in the type, so no
operator can break it by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable

    from fleetlab.domain.ids import RequestId, VehicleId
    from fleetlab.domain.solution import Solution
    from fleetlab.study import Study


@runtime_checkable
class Move(Protocol):
    """A single step from one solution to another."""

    @property
    def name(self) -> str:
        """Operator name, for adaptive selection and run logs."""
        ...

    @property
    def touched(self) -> tuple[VehicleId, ...]:
        """Vehicles whose routes this move changes. Drives delta evaluation."""
        ...

    def apply(self, solution: Solution) -> Solution:
        """The solution that results. Does not modify ``solution``."""
        ...


@dataclass(frozen=True, slots=True)
class MoveEvaluation:
    """What a move would cost and whether it would be legal.

    Attributes:
        move: The move evaluated.
        delta: Change in objective value. Negative is an improvement.
        feasible: Whether the resulting solution satisfies every rule.
        penalty: Weighted violation total of the result, zero when feasible.
    """

    move: Move
    delta: float
    feasible: bool
    penalty: float = 0.0

    @property
    def improves(self) -> bool:
        """Whether this move lowers cost without breaking anything."""
        return self.feasible and self.delta < 0.0

    def score(self, penalty_weight: float = 1.0) -> float:
        """Delta plus weighted penalty, for searches that accept infeasible steps."""
        return self.delta + penalty_weight * self.penalty

    def __str__(self) -> str:
        mark = "ok" if self.feasible else f"penalty {self.penalty:.4g}"
        return f"{self.move.name} delta={self.delta:+.4f} ({mark})"


def delta_cost(study: Study, solution: Solution, move: Move) -> float:
    """Change in objective value from applying a move, re-costing only what changed.

    Fleet-level terms -- an unserved-request charge, for instance -- are cheap to
    recompute in full, so they are, and the per-route terms are recomputed only
    for the vehicles the move touches.
    """
    after = move.apply(solution)
    before_routes = sum(study.route_cost(solution.route_for(v)) for v in move.touched)
    after_routes = sum(study.route_cost(after.route_for(v)) for v in move.touched)

    objective = study.objective
    before_fleet = sum(
        objective.weight_of(term.name)
        * term.fleet_cost(solution, study.timings(solution), study.context)
        for term in objective.terms
    )
    after_fleet = sum(
        objective.weight_of(term.name) * term.fleet_cost(after, study.timings(after), study.context)
        for term in objective.terms
    )
    return (after_routes - before_routes) + (after_fleet - before_fleet)


def evaluate_move(study: Study, solution: Solution, move: Move) -> MoveEvaluation:
    """Cost and legality of a move, without committing to it."""
    after = move.apply(solution)
    delta = delta_cost(study, solution, move)
    report = study.check(after)
    return MoveEvaluation(
        move=move,
        delta=delta,
        feasible=report.feasible,
        penalty=report.penalty(),
    )


def best_move(
    study: Study,
    solution: Solution,
    moves: Iterable[Move],
    *,
    allow_infeasible: bool = False,
    penalty_weight: float = 1.0,
) -> MoveEvaluation | None:
    """The lowest-scoring move from a set, or ``None`` if none qualifies.

    Args:
        study: The study to cost against.
        solution: The incumbent.
        moves: Candidates.
        allow_infeasible: Whether infeasible results may be returned, scored
            with their penalty.
        penalty_weight: Multiplier on violation magnitude when they may.
    """
    best: MoveEvaluation | None = None
    for move in moves:
        evaluation = evaluate_move(study, solution, move)
        if not evaluation.feasible and not allow_infeasible:
            continue
        score = evaluation.score(penalty_weight)
        if best is None or score < best.score(penalty_weight):
            best = evaluation
    return best


@dataclass(frozen=True, slots=True)
class ReleaseRequests:
    """Take requests off their routes and back into the unassigned pool.

    The ruin half of ruin-and-recreate.

    Attributes:
        requests: What to release.
        from_vehicles: Which routes will change. Supplied by the operator that
            built the move, so delta evaluation need not search for them.
    """

    requests: tuple[RequestId, ...]
    from_vehicles: tuple[VehicleId, ...]

    @property
    def name(self) -> str:
        """Operator name."""
        return "release"

    @property
    def touched(self) -> tuple[VehicleId, ...]:
        """Routes this move changes."""
        return self.from_vehicles

    def apply(self, solution: Solution) -> Solution:
        """Strip the requests from every route and add them to the pool."""
        return solution.with_released(self.requests)
