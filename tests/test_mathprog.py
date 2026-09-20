"""The mathematical-programming lane, and its agreement with the search lane.

The central test is :func:`test_solver_optimum_agrees_with_the_evaluator`. It is
what the whole two-faced-constraint design exists to make possible: solve the
model, read the answer back as an ordinary
:class:`~fleetlab.domain.solution.Solution`, and evaluate it with the *same*
evaluator, constraints and objective every heuristic uses. If the model and the
evaluator disagree about the problem, that test fails -- and without it, the
disagreement would show up only as a bound that quietly does not bound anything.
"""

from __future__ import annotations

import pytest

from fleetlab.io.generate import tiny_instance
from fleetlab.linear import Direction, LinExpr, Model, Sense, VarKind, equals
from fleetlab.mathprog import assignment_to_solution, build_model, solution_to_assignment
from fleetlab.od import PiecewiseODMatrix
from fleetlab.search import RegretInsertion
from fleetlab.study import Study

pulp = pytest.importorskip("pulp", reason="the solvers extra is not installed")


# --------------------------------------------------------------- the algebra


def test_linear_expression_arithmetic() -> None:
    model = Model()
    x = model.add_var("x")
    y = model.add_var("y")

    expression = 2 * x + 3 * y - x + 5
    assert expression.terms[x.index] == pytest.approx(1.0)
    assert expression.terms[y.index] == pytest.approx(3.0)
    assert expression.constant == pytest.approx(5.0)


def test_zero_coefficients_are_dropped() -> None:
    model = Model()
    x = model.add_var("x")
    assert (x - x).terms == {}


def test_row_moves_variables_left_and_constants_right() -> None:
    model = Model()
    x = model.add_var("x")
    y = model.add_var("y")
    row = model.add(x + 3 >= y + 10)
    assert row.sense is Sense.GE
    assert row.lhs.terms[x.index] == pytest.approx(1.0)
    assert row.lhs.terms[y.index] == pytest.approx(-1.0)
    assert row.rhs == pytest.approx(7.0)


def test_equality_uses_a_function_not_an_operator() -> None:
    """``==`` is not overloaded, so an accidental comparison cannot become a row."""
    model = Model()
    x = model.add_var("x")
    assert equals(x, 4.0).sense is Sense.EQ
    assert (x == x) is True  # ordinary object comparison, not a constraint


def test_duplicate_variable_names_are_refused() -> None:
    model = Model()
    model.add_var("x")
    with pytest.raises(ValueError, match="Duplicate variable name"):
        model.add_var("x")


def test_lp_export_has_the_expected_sections() -> None:
    model = Model()
    x = model.add_var("x", kind=VarKind.BINARY)
    y = model.add_var("y", lower=1.0, upper=8.0)
    model.add(x + y <= 6.0, "cap")
    model.minimise(LinExpr.sum([2 * x, y]))

    text = model.to_lp()
    assert text.startswith("Minimize")
    assert "Subject To" in text
    assert "cap:" in text
    assert "Bounds" in text
    assert "Binaries" in text
    assert text.rstrip().endswith("End")
    assert model.direction is Direction.MINIMISE


# ------------------------------------------------------------- the formulation


def test_model_builds_and_reports_itself_exact() -> None:
    study = Study(tiny_instance())
    report = build_model(study)

    assert report.model.variable_count(VarKind.BINARY) > 0
    assert report.model.row_count > 0
    assert report.is_exact, report.describe()
    assert report.dropped_constraints == ()
    assert report.dropped_objective_terms == ()


def test_arc_pruning_removes_time_infeasible_arcs() -> None:
    study = Study(tiny_instance())
    pruned = build_model(study, prune_arcs=True)
    whole = build_model(study, prune_arcs=False)
    assert pruned.arcs_kept < whole.arcs_kept
    assert pruned.arcs_considered == whole.arcs_considered


def test_time_dependent_matrix_is_refused_without_an_explicit_decision() -> None:
    """Freezing a travel profile must be an explicit, recorded choice.

    A bound computed against a frozen profile is not a bound for the true
    problem, so the exporter must never do it silently.
    """
    base = tiny_instance()
    stops = list(base.stops.values())
    profile = PiecewiseODMatrix(
        boundaries=[0.0, 120.0],
        durations={
            (a.id, b.id): [
                base.od.duration(0.0, a.id, b.id),
                base.od.duration(0.0, a.id, b.id) * 1.5,
            ]
            for a in stops
            for b in stops
            if a.id != b.id
        },
        distances={
            (a.id, b.id): base.od.distance(a.id, b.id) for a in stops for b in stops if a.id != b.id
        },
    )
    from fleetlab.domain import Problem

    dependent = Problem.build(
        name="dependent",
        capacity_space=base.capacity_space,
        stops=stops,
        requests=base.requests.values(),
        vehicles=base.vehicles.values(),
        od=profile,
        horizon_end=base.horizon_end,
    )
    study = Study(dependent)

    with pytest.raises(ValueError, match="freeze_profile_at"):
        build_model(study)

    approximate = build_model(study, freeze_profile_at=60.0)
    assert not approximate.is_exact
    assert approximate.frozen_profile_at == pytest.approx(60.0)
    assert "not a valid bound" in approximate.describe()


def test_freeze_instant_is_refused_for_a_constant_matrix() -> None:
    study = Study(tiny_instance())
    with pytest.raises(ValueError, match="matrix is constant"):
        build_model(study, freeze_profile_at=10.0)


# ---------------------------------------------------------------- round trip


def test_solution_round_trips_through_a_variable_assignment() -> None:
    study = Study(tiny_instance())
    report = build_model(study)
    built = RegretInsertion().solve(study).solution

    assignment = solution_to_assignment(study, report.variables, built)
    recovered = assignment_to_solution(study, report.variables, assignment)

    assert recovered.routes == built.routes
    assert recovered.unassigned == built.unassigned
    assert study.cost(recovered) == pytest.approx(study.cost(built))


# ----------------------------------------------------- the cross-lane agreement


@pytest.mark.slow
def test_solver_optimum_agrees_with_the_evaluator() -> None:
    """Solve the model, read the answer back, and re-evaluate it independently.

    The solver's objective value and the evaluator's cost for the same solution
    must agree. They are computed by entirely separate code from the same
    constraint and objective objects, so agreement is real evidence that the
    two lanes state one problem.
    """
    from fleetlab.mathprog.adapters import solve_with_pulp

    study = Study(tiny_instance())
    report = build_model(study)
    warm = solution_to_assignment(study, report.variables, RegretInsertion().solve(study).solution)

    result = solve_with_pulp(report.model, time_limit=180, warm_start=warm)
    assert result.has_solution, f"solver returned {result.status}"

    recovered = assignment_to_solution(study, report.variables, result.values)
    feasibility = study.check(recovered)

    assert feasibility.feasible, feasibility.describe()
    assert result.objective is not None
    assert study.cost(recovered) == pytest.approx(result.objective, rel=1e-6)


@pytest.mark.slow
def test_solver_optimum_is_no_worse_than_the_heuristic() -> None:
    """A proved optimum must not be beaten by a construction heuristic.

    If it is, the model is a relaxation of something other than the study's
    problem, or the objective renders differently in the two lanes.
    """
    from fleetlab.mathprog.adapters import solve_with_pulp

    study = Study(tiny_instance())
    report = build_model(study)
    heuristic = RegretInsertion().solve(study)
    warm = solution_to_assignment(study, report.variables, heuristic.solution)

    result = solve_with_pulp(report.model, time_limit=180, warm_start=warm)
    if not result.optimal:
        pytest.skip(f"solver did not prove optimality ({result.status})")

    assert result.objective is not None
    assert result.objective <= heuristic.cost + 1e-6
