"""Constraints report magnitudes, and never raise.

The two properties the design rests on:

* infeasibility comes back as data with a size, so a penalty-based search can
  steer on it;
* checking never raises and never mutates, so it can run on a candidate that
  was never installed anywhere.
"""

from __future__ import annotations

import pytest

from fleetlab.domain import (
    ActionType,
    Problem,
    RequestId,
    Schedule,
    Solution,
    VehicleId,
)
from fleetlab.feasibility import (
    AllRequestsServed,
    CapacityLimit,
    MaxOnboardTime,
    Pairing,
    Precedence,
    TimeWindows,
    VehicleAvailability,
    standard_constraints,
)
from fleetlab.study import Study
from fleetlab.timing import EvalContext, evaluate_route, evaluate_solution


def test_feasible_route_reports_nothing(line_study: Study, line_problem: Problem) -> None:
    p1, d1 = line_problem.actions_for(RequestId("r1"))
    report = line_study.check_route(Schedule(VehicleId("v1"), (p1, d1)))
    assert report.feasible
    assert report.penalty() == pytest.approx(0.0)


def test_precedence_violation_has_a_magnitude(line_problem: Problem) -> None:
    p1, d1 = line_problem.actions_for(RequestId("r1"))
    inverted = Schedule(VehicleId("v1"), (d1, p1))
    ctx = EvalContext(line_problem)
    timing = evaluate_route(inverted, ctx)

    violations = list(Precedence().check_route(inverted, timing, ctx))
    assert len(violations) == 1
    assert violations[0].constraint == "precedence"
    assert violations[0].magnitude > 0
    assert violations[0].unit == "positions"


def test_pairing_catches_a_half_placed_request(line_problem: Problem) -> None:
    p1, _ = line_problem.actions_for(RequestId("r1"))
    half = Schedule(VehicleId("v1"), (p1,))
    ctx = EvalContext(line_problem)
    violations = list(Pairing().check_route(half, evaluate_route(half, ctx), ctx))
    assert [v.constraint for v in violations] == ["pairing"]
    assert violations[0].request == RequestId("r1")


def test_capacity_reports_the_offending_dimension(line_problem: Problem) -> None:
    """A vehicle with no seats must reject a passenger and say which dimension."""
    space = line_problem.capacity_space
    vehicle = line_problem.vehicle(VehicleId("v1"))
    seatless = Problem.build(
        name="seatless",
        capacity_space=space,
        stops=line_problem.stops.values(),
        requests=line_problem.requests.values(),
        vehicles=[
            type(vehicle)(
                id=vehicle.id,
                start_stop=vehicle.start_stop,
                end_stop=vehicle.end_stop,
                capacity=space.of(seats=0, kg=500.0),
                available_from=vehicle.available_from,
                available_until=vehicle.available_until,
            )
        ],
        od=line_problem.od,
        horizon_end=line_problem.horizon_end,
    )
    p1, d1 = seatless.actions_for(RequestId("r1"))
    route = Schedule(VehicleId("v1"), (p1, d1))
    ctx = EvalContext(seatless)

    violations = list(CapacityLimit().check_route(route, evaluate_route(route, ctx), ctx))
    assert len(violations) == 1
    assert violations[0].unit == "seats"
    assert violations[0].magnitude == pytest.approx(1.0)


def test_goods_only_vehicle_still_carries_parcels(line_problem: Problem) -> None:
    """The people/goods split must fall out of capacity, not a type check."""
    space = line_problem.capacity_space
    vehicle = line_problem.vehicle(VehicleId("v1"))
    seatless = Problem.build(
        name="seatless",
        capacity_space=space,
        stops=line_problem.stops.values(),
        requests=line_problem.requests.values(),
        vehicles=[
            type(vehicle)(
                id=vehicle.id,
                start_stop=vehicle.start_stop,
                end_stop=vehicle.end_stop,
                capacity=space.of(seats=0, kg=500.0),
                available_from=vehicle.available_from,
                available_until=vehicle.available_until,
            )
        ],
        od=line_problem.od,
        horizon_end=line_problem.horizon_end,
    )
    p2, d2 = seatless.actions_for(RequestId("r2"))
    route = Schedule(VehicleId("v1"), (p2, d2))
    ctx = EvalContext(seatless)
    assert not list(CapacityLimit().check_route(route, evaluate_route(route, ctx), ctx))


def test_time_window_violation_is_measured_in_minutes(line_problem: Problem) -> None:
    p1, d1 = line_problem.actions_for(RequestId("r1"))
    p2, d2 = line_problem.actions_for(RequestId("r2"))
    # Serve r2 first and far away, so r1's pickup misses its 40:00 close.
    late = Schedule(VehicleId("v1"), (p2, d2, p1, d1))
    ctx = EvalContext(line_problem)
    timing = evaluate_route(late, ctx)

    violations = [
        violation
        for violation in TimeWindows().check_route(late, timing, ctx)
        if violation.request == RequestId("r1")
    ]
    assert violations
    assert violations[0].unit == "minutes"
    window = line_problem.request(RequestId("r1")).window_for(ActionType.PICKUP)
    position = late.position_of(RequestId("r1"), ActionType.PICKUP)
    assert position is not None
    expected = timing.stops[position].service_start - window.latest
    assert violations[0].magnitude == pytest.approx(expected)


def test_max_onboard_time_is_optional_per_request(line_problem: Problem) -> None:
    """r2 states no cap, so it can never violate one -- that is what makes the.

    framework usable for goods without a separate code path.
    """
    p2, d2 = line_problem.actions_for(RequestId("r2"))
    p1, d1 = line_problem.actions_for(RequestId("r1"))
    detoured = Schedule(VehicleId("v1"), (p2, p1, d1, d2))
    ctx = EvalContext(line_problem)
    timing = evaluate_route(detoured, ctx)

    reported = [v.request for v in MaxOnboardTime().check_route(detoured, timing, ctx)]
    assert RequestId("r2") not in reported


def test_max_onboard_time_default_limit_applies_to_uncapped_requests(
    line_problem: Problem,
) -> None:
    p2, d2 = line_problem.actions_for(RequestId("r2"))
    p1, d1 = line_problem.actions_for(RequestId("r1"))
    detoured = Schedule(VehicleId("v1"), (p2, p1, d1, d2))
    ctx = EvalContext(line_problem)
    timing = evaluate_route(detoured, ctx)

    strict = MaxOnboardTime(default_limit=1.0)
    reported = [v.request for v in strict.check_route(detoured, timing, ctx)]
    assert RequestId("r2") in reported


def test_availability_counts_the_return_leg(line_problem: Problem) -> None:
    """The Java original stopped at the last action; a vehicle that cannot get.

    home in time counted as feasible.
    """
    space = line_problem.capacity_space
    vehicle = line_problem.vehicle(VehicleId("v1"))
    short_shift = Problem.build(
        name="short",
        capacity_space=space,
        stops=line_problem.stops.values(),
        requests=line_problem.requests.values(),
        vehicles=[
            type(vehicle)(
                id=vehicle.id,
                start_stop=vehicle.start_stop,
                end_stop=vehicle.end_stop,
                capacity=vehicle.capacity,
                available_from=0.0,
                # Serving r1 alone: leave at 17:00, depart c at 30:00, and the
                # return leg from c to the depot is 9 minutes, so the vehicle is
                # home at 39:00 -- four minutes past this shift end.
                available_until=35.0,
            )
        ],
        od=line_problem.od,
        horizon_end=300.0,
    )
    p1, d1 = short_shift.actions_for(RequestId("r1"))
    route = Schedule(VehicleId("v1"), (p1, d1))
    ctx = EvalContext(short_shift)
    timing = evaluate_route(route, ctx)

    assert timing.return_arrival == pytest.approx(39.0)
    violations = list(VehicleAvailability().check_route(route, timing, ctx))
    assert violations, "the return leg must be counted against the shift end"
    assert violations[0].magnitude == pytest.approx(4.0)


def test_coverage_is_a_fleet_level_rule(line_problem: Problem) -> None:
    ctx = EvalContext(line_problem)
    solution = Solution((Schedule(VehicleId("v1")),), frozenset({RequestId("r1"), RequestId("r2")}))
    timings = evaluate_solution(solution, ctx)
    violations = list(AllRequestsServed().check_solution(solution, timings, ctx))
    assert len(violations) == 2
    assert all(v.unit == "requests" for v in violations)


def test_penalty_weights_are_per_constraint(line_problem: Problem) -> None:
    p1, d1 = line_problem.actions_for(RequestId("r1"))
    inverted = Schedule(VehicleId("v1"), (d1, p1))
    study = Study(line_problem)
    report = study.check_route(inverted)

    plain = report.penalty()
    weighted = report.penalty({"precedence": 1000.0})
    assert weighted > plain


def test_checking_never_raises_on_a_nonsensical_route(line_problem: Problem) -> None:
    """Every rule must survive garbage.

    Raising here would make a penalised
    search impossible, since it must evaluate infeasible candidates.
    """
    p1, d1 = line_problem.actions_for(RequestId("r1"))
    p2, d2 = line_problem.actions_for(RequestId("r2"))
    nonsense = Schedule(VehicleId("v1"), (d1, d2, p1, p2, p1))
    study = Study(line_problem)
    report = study.check_route(nonsense)
    assert not report.feasible
    assert report.penalty() > 0


def test_constraint_set_composition(line_problem: Problem) -> None:
    del line_problem
    full = standard_constraints()
    assert "max_onboard_time" in full.names()
    goods = standard_constraints(max_onboard_time=False)
    assert "max_onboard_time" not in goods.names()
    trimmed = full.without("capacity")
    assert "capacity" not in trimmed.names()


def test_every_standard_constraint_is_linearisable() -> None:
    """If a rule cannot be rendered into a model, the two lanes are solving.

    different problems -- so the set must say so loudly.
    """
    assert standard_constraints().non_linearisable() == ()
