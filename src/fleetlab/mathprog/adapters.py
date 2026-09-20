"""Optional in-process solving.

The framework's mathematical-programming lane has **no solver dependency**: a
study can export an LP file and hand it to whatever it has. This module is for
studies that would rather solve in process, and it is deliberately the only
place that imports a solver.

Install with ``pip install fleetlab[solvers]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fleetlab.linear import Direction, Sense, VarKind

if TYPE_CHECKING:
    from fleetlab.linear import Model


@dataclass(frozen=True, slots=True)
class SolveResult:
    """What a solver returned.

    Attributes:
        status: The solver's own status string, verbatim.
        objective: Objective value, if one was found.
        values: Variable name to value.
        optimal: Whether the solver proved optimality. A feasible-but-not-proven
            answer is still useful; quoting it as an optimum is not.
    """

    status: str
    objective: float | None
    values: dict[str, float]
    optimal: bool

    @property
    def has_solution(self) -> bool:
        """Whether any variable values came back."""
        return bool(self.values)


def write_lp(model: Model, path: str | Path) -> Path:
    """Write a model to an LP file.

    The zero-dependency route: CBC, HiGHS, SCIP, Gurobi and CPLEX all read this.
    """
    destination = Path(path)
    destination.write_text(model.to_lp(), encoding="utf-8")
    return destination


def solve_with_pulp(
    model: Model,
    *,
    time_limit: float | None = None,
    warm_start: dict[str, float] | None = None,
    msg: bool = False,
) -> SolveResult:
    """Solve a model with PuLP's bundled CBC.

    Args:
        model: The program to solve.
        time_limit: Seconds. ``None`` means no limit, which on a
            pickup-and-delivery model of any size means "until the heat death of
            the universe" -- set one.
        warm_start: Initial variable values, as
            :func:`~fleetlab.mathprog.roundtrip.solution_to_assignment` produces.
        msg: Whether to let the solver print to stdout.

    Returns:
        A :class:`SolveResult`.

    Raises:
        ImportError: If the ``solvers`` extra is not installed.
    """
    try:
        import pulp
    except ImportError as error:  # pragma: no cover -- exercised only without the extra
        msg_text = (
            "solve_with_pulp needs the optional solver dependency. "
            "Install it with: pip install 'fleetlab[solvers]'. "
            "Alternatively use write_lp() and solve with any external solver."
        )
        raise ImportError(msg_text) from error

    sense = pulp.LpMinimize if model.direction is Direction.MINIMISE else pulp.LpMaximize
    program = pulp.LpProblem("fleetlab", sense)

    lookup: dict[int, pulp.LpVariable] = {}
    for variable in model.variables:
        category = {
            VarKind.BINARY: pulp.LpBinary,
            VarKind.INTEGER: pulp.LpInteger,
            VarKind.CONTINUOUS: pulp.LpContinuous,
        }[variable.kind]
        built = pulp.LpVariable(
            variable.name,
            lowBound=None if variable.lower == float("-inf") else variable.lower,
            upBound=None if variable.upper == float("inf") else variable.upper,
            cat=category,
        )
        if warm_start is not None and variable.name in warm_start:
            built.setInitialValue(warm_start[variable.name])
        lookup[variable.index] = built

    program += (
        pulp.lpSum(
            coefficient * lookup[index] for index, coefficient in model.objective.terms.items()
        )
        + model.objective.constant
    )

    for row in model.rows:
        expression = pulp.lpSum(
            coefficient * lookup[index] for index, coefficient in row.lhs.terms.items()
        )
        if row.sense is Sense.LE:
            program += (expression <= row.rhs, row.name)
        elif row.sense is Sense.GE:
            program += (expression >= row.rhs, row.name)
        else:
            program += (expression == row.rhs, row.name)

    solver = pulp.PULP_CBC_CMD(
        msg=msg,
        timeLimit=time_limit,
        warmStart=warm_start is not None,
    )
    program.solve(solver)

    status = pulp.LpStatus[program.status]
    values = {
        variable.name: float(lookup[variable.index].value() or 0.0) for variable in model.variables
    }
    objective = pulp.value(program.objective)
    return SolveResult(
        status=status,
        objective=float(objective) if objective is not None else None,
        values=values,
        optimal=status == "Optimal",
    )
