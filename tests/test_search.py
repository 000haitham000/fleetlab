"""Search algorithms and the objective they optimise."""

from __future__ import annotations

import pytest

from fleetlab.io.generate import mixed_instance
from fleetlab.objective import (
    ExcessOnboardTime,
    FleetSize,
    Objective,
    TravelDistance,
    UnservedPenalty,
)
from fleetlab.search import AdaptiveLNS, CheapestInsertion, RegretInsertion
from fleetlab.study import Study


@pytest.fixture(scope="module")
def instance() -> object:
    return mixed_instance(passengers=6, wheelchair_users=2, parcels=5, vehicles=3, seed=17)


def test_construction_produces_feasible_routes(instance: object) -> None:
    study = Study(instance)  # type: ignore[arg-type]
    result = RegretInsertion().solve(study)

    for schedule in result.solution.routes:
        assert study.check_route(schedule).feasible, study.check_route(schedule).describe()


def test_every_assigned_request_has_both_ends_on_one_route(instance: object) -> None:
    study = Study(instance)  # type: ignore[arg-type]
    result = RegretInsertion().solve(study)

    for request_id in result.solution.assigned_requests():
        vehicle = result.solution.vehicle_of(request_id)
        assert vehicle is not None
        positions = result.solution.route_for(vehicle).positions_of(request_id)
        assert positions is not None, f"{request_id} is only half placed"
        assert positions[0] < positions[1]


def test_construction_is_deterministic(instance: object) -> None:
    first = RegretInsertion().solve(Study(instance))  # type: ignore[arg-type]
    second = RegretInsertion().solve(Study(instance))  # type: ignore[arg-type]
    assert first.cost == pytest.approx(second.cost)
    assert first.solution.routes == second.solution.routes


def test_slack_screen_does_not_change_the_answer(instance: object) -> None:
    """The screen is an optimisation, so it must be invisible in the result."""
    screened = RegretInsertion(use_slack_screen=True).solve(Study(instance))  # type: ignore[arg-type]
    plain = RegretInsertion(use_slack_screen=False).solve(Study(instance))  # type: ignore[arg-type]
    assert screened.cost == pytest.approx(plain.cost)
    assert screened.solution.routes == plain.solution.routes


def test_lns_never_returns_worse_than_its_start(instance: object) -> None:
    study = Study(instance)  # type: ignore[arg-type]
    start = RegretInsertion().solve(study)
    improved = AdaptiveLNS(iterations=120, seed=5).solve(study, start.solution)
    assert improved.cost <= start.cost + 1e-9


def test_lns_is_reproducible(instance: object) -> None:
    """A result nobody can re-run is not a result."""
    first = AdaptiveLNS(iterations=80, seed=99).solve(Study(instance))  # type: ignore[arg-type]
    second = AdaptiveLNS(iterations=80, seed=99).solve(Study(instance))  # type: ignore[arg-type]
    assert first.cost == pytest.approx(second.cost)


def test_different_seeds_explore_differently(instance: object) -> None:
    first = AdaptiveLNS(iterations=80, seed=1).solve(Study(instance))  # type: ignore[arg-type]
    second = AdaptiveLNS(iterations=80, seed=2).solve(Study(instance))  # type: ignore[arg-type]
    assert first.notes["seed"] != second.notes["seed"]


def test_run_result_reports_more_than_a_number(instance: object) -> None:
    result = RegretInsertion().solve(Study(instance))  # type: ignore[arg-type]
    assert result.iterations > 0
    assert result.evaluations > 0
    assert result.history
    assert "cost" in result.describe()


def test_cost_breakdown_sums_to_the_total(instance: object) -> None:
    study = Study(instance)  # type: ignore[arg-type]
    solution = RegretInsertion().solve(study).solution
    breakdown = study.breakdown(solution)
    assert sum(breakdown.terms.values()) == pytest.approx(breakdown.total)
    assert breakdown.total == pytest.approx(study.cost(solution))


def test_reweighting_changes_the_trade_off(instance: object) -> None:
    objective = Objective(
        terms=(TravelDistance(), ExcessOnboardTime(), FleetSize(), UnservedPenalty()),
        weights={"fleet_size": 1.0, "unserved": 10_000.0},
    )
    cheap_fleet = Study(instance, objective=objective)  # type: ignore[arg-type]
    dear_fleet = Study(instance, objective=objective.reweighted(fleet_size=50_000.0))  # type: ignore[arg-type]

    lenient = RegretInsertion().solve(cheap_fleet).solution
    strict = RegretInsertion().solve(dear_fleet).solution
    assert len(strict.used_vehicles()) <= len(lenient.used_vehicles())


def test_objective_rejects_duplicate_term_names() -> None:
    with pytest.raises(ValueError, match="duplicate term names"):
        Objective(terms=(TravelDistance(), TravelDistance()))


def test_cheapest_and_regret_both_terminate(instance: object) -> None:
    for algorithm in (CheapestInsertion(), RegretInsertion(k=2), RegretInsertion(k=4)):
        result = algorithm.solve(Study(instance))  # type: ignore[arg-type]
        assert result.iterations <= len(instance.requests)  # type: ignore[attr-defined]
