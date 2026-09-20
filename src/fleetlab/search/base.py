"""The algorithm interface, and what a run reports.

Every algorithm in the framework -- a construction heuristic, a metaheuristic, a
solver wrapper -- implements one method: given a :class:`~fleetlab.study.Study`
and a starting solution, return a better one. That uniformity is what makes a
comparison table possible.

:class:`RunResult` is deliberately richer than "here is the answer". A study
comparing algorithms needs to know how long each took, how many candidates it
looked at, and how its incumbent improved over time -- otherwise the comparison
reduces to a single number per algorithm, which hides every interesting
difference between them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fleetlab.domain.solution import Solution
    from fleetlab.feasibility.violation import Feasibility
    from fleetlab.objective.base import CostBreakdown
    from fleetlab.study import Study


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """The incumbent at one moment of a run.

    Attributes:
        iteration: Which iteration produced it.
        elapsed: Seconds since the run began.
        cost: Objective value.
        feasible: Whether it broke any rule.
        unassigned: How many requests were still unplaced.
    """

    iteration: int
    elapsed: float
    cost: float
    feasible: bool
    unassigned: int


@dataclass(frozen=True, slots=True)
class RunResult:
    """What an algorithm produced, and what it took to produce it.

    Attributes:
        solution: The best solution found.
        cost: Its objective value.
        breakdown: Per-term figures behind that value.
        feasibility: What, if anything, it breaks.
        iterations: How many iterations ran.
        elapsed: Wall-clock seconds.
        evaluations: Candidate evaluations performed, where the algorithm counts
            them. The honest denominator for a like-for-like comparison, since
            wall-clock time confounds algorithm quality with implementation
            effort.
        history: Incumbent checkpoints over the run.
        notes: Free-form per-algorithm detail for the results table.
    """

    solution: Solution
    cost: float
    breakdown: CostBreakdown
    feasibility: Feasibility
    iterations: int = 0
    elapsed: float = 0.0
    evaluations: int = 0
    history: tuple[Checkpoint, ...] = ()
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        """Whether the returned solution satisfies every rule."""
        return self.feasibility.feasible

    def describe(self) -> str:
        """A multi-line report for a run log."""
        status = "feasible" if self.feasible else f"{len(self.feasibility)} violations"
        lines = [
            f"cost {self.cost:.4f} ({status})",
            f"{self.iterations} iterations, {self.evaluations} evaluations, {self.elapsed:.2f}s",
            f"{len(self.solution.used_vehicles())} vehicles used, "
            f"{len(self.solution.unassigned)} unassigned",
            self.breakdown.describe(),
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return f"RunResult(cost={self.cost:.4f}, feasible={self.feasible})"


@runtime_checkable
class Algorithm(Protocol):
    """Something that improves a solution to a study."""

    @property
    def name(self) -> str:
        """Identifier for the results table."""
        ...

    def solve(self, study: Study, initial: Solution | None = None) -> RunResult:
        """Produce the best solution it can.

        Args:
            study: The instance, rules and objective.
            initial: A starting solution. ``None`` means start from empty, which
                is what a construction heuristic expects; a metaheuristic given
                ``None`` should construct one first rather than fail.
        """
        ...


class RunRecorder:
    """Bookkeeping shared by algorithm implementations.

    Keeps the timing, counting and checkpoint logic in one place so that two
    algorithms' reported numbers mean the same thing -- which they will not if
    each one counts its own way.
    """

    __slots__ = ("_evaluations", "_history", "_iterations", "_started")

    def __init__(self) -> None:
        self._started = time.perf_counter()
        self._iterations = 0
        self._evaluations = 0
        self._history: list[Checkpoint] = []

    @property
    def elapsed(self) -> float:
        """Seconds since the recorder was created."""
        return time.perf_counter() - self._started

    @property
    def iterations(self) -> int:
        """Iterations counted so far."""
        return self._iterations

    @property
    def evaluations(self) -> int:
        """Candidate evaluations counted so far."""
        return self._evaluations

    def tick(self, count: int = 1) -> None:
        """Count iterations."""
        self._iterations += count

    def count_evaluations(self, count: int) -> None:
        """Count candidate evaluations."""
        self._evaluations += count

    def checkpoint(self, study: Study, solution: Solution) -> None:
        """Record the incumbent."""
        report = study.check(solution)
        self._history.append(
            Checkpoint(
                iteration=self._iterations,
                elapsed=self.elapsed,
                cost=study.cost(solution),
                feasible=report.feasible,
                unassigned=len(solution.unassigned),
            )
        )

    def history(self) -> Sequence[Checkpoint]:
        """Every checkpoint recorded."""
        return tuple(self._history)

    def finish(
        self,
        study: Study,
        solution: Solution,
        notes: dict[str, str] | None = None,
    ) -> RunResult:
        """Assemble the result."""
        return RunResult(
            solution=solution,
            cost=study.cost(solution),
            breakdown=study.breakdown(solution),
            feasibility=study.check(solution),
            iterations=self._iterations,
            elapsed=self.elapsed,
            evaluations=self._evaluations,
            history=tuple(self._history),
            notes=notes or {},
        )
