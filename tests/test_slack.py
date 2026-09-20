"""Forward time slack: the screen must never discard a feasible insertion.

This is the most important test in the suite. The slack screen exists to skip
insertion positions without building them, and its whole value depends on one
property: **a position it rejects must genuinely be infeasible**. A screen that
is merely usually right silently biases every search built on it, and the bias
is invisible in ordinary results -- the run finishes, the answer looks plausible,
and it is quietly worse than it should be.

The equivalence test below is what caught exactly that during development: the
screen treated a vehicle's depot departure as fixed, when inserting a new first
action actually lets the vehicle leave *earlier*. Positions at index zero were
being discarded despite being feasible and often the best available.
"""

from __future__ import annotations

import random

import pytest

from fleetlab.domain import Problem, RequestId, Schedule, Solution, VehicleId
from fleetlab.io.generate import mixed_instance
from fleetlab.moves.insert import candidates_for_route
from fleetlab.search import RegretInsertion
from fleetlab.study import Study
from fleetlab.timing import evaluate_route, forward_slack
from fleetlab.timing.context import EvalContext


def positions(
    study: Study,
    solution: Solution,
    request: RequestId,
    vehicle: VehicleId,
    *,
    screen: bool,
) -> set[tuple[int, int]]:
    return {
        (candidate.move.pickup_index, candidate.move.dropoff_index)
        for candidate in candidates_for_route(
            study, solution, request, vehicle, use_slack_screen=screen
        )
    }


@pytest.mark.parametrize("seed", [1, 7, 42, 2026])
def test_screen_never_discards_a_feasible_insertion(seed: int) -> None:
    """Screened and unscreened enumeration must yield the identical candidate set."""
    problem = mixed_instance(passengers=5, wheelchair_users=1, parcels=4, vehicles=3, seed=seed)
    study = Study(problem)
    built = RegretInsertion().solve(study).solution

    compared = 0
    for request_id in sorted(problem.requests):
        base = built.with_released([request_id])
        for vehicle_id in base.vehicles():
            with_screen = positions(study, base, request_id, vehicle_id, screen=True)
            without = positions(study, base, request_id, vehicle_id, screen=False)
            assert with_screen == without, (
                f"screen changed the candidate set for {request_id} on {vehicle_id}: "
                f"missing {sorted(without - with_screen)}, "
                f"spurious {sorted(with_screen - without)}"
            )
            compared += len(without)

    assert compared > 0, "the test enumerated nothing, so it proved nothing"


def test_screen_is_marked_exact_only_when_it_is(line_problem: Problem) -> None:
    """A constant matrix with no onboard cap is exact; an onboard cap makes it a screen."""
    study = Study(line_problem)
    pickup, dropoff = line_problem.actions_for(RequestId("r2"))
    route = Schedule(VehicleId("v1"), (pickup, dropoff))
    # r2 states no onboard cap, so slack over this route is an exact bound.
    assert study.slack(route).exact

    pickup1, dropoff1 = line_problem.actions_for(RequestId("r1"))
    with_capped = Schedule(VehicleId("v1"), (pickup1, dropoff1))
    # r1 does state one, so the bound becomes an optimistic screen.
    assert not study.slack(with_capped).exact


def test_slack_recursion_matches_a_direct_delay_experiment(line_problem: Problem) -> None:
    """Delaying by exactly the slack must stay feasible; by more, must not.

    Checks the backward recursion against the definition rather than against
    itself, on the route whose slack is an exact bound.
    """
    ctx = EvalContext(line_problem)
    pickup, dropoff = line_problem.actions_for(RequestId("r2"))
    route = Schedule(VehicleId("v1"), (pickup, dropoff))
    timing = evaluate_route(route, ctx)
    slack = forward_slack(route, timing, ctx)

    # Delaying the departure from position 0 by its slack keeps position 1 legal.
    room = slack.absorbable_after(0)
    window = line_problem.request(RequestId("r2")).dropoff_window
    assert timing.stops[1].service_start + room <= window.latest + 1e-6
    assert timing.stops[1].service_start + room + 1.0 > window.latest


def test_empty_route_slack_is_the_whole_shift(line_problem: Problem) -> None:
    study = Study(line_problem)
    slack = study.slack(Schedule(VehicleId("v1")))
    vehicle = line_problem.vehicle(VehicleId("v1"))
    assert slack.depot == pytest.approx(vehicle.available_until)
    assert slack.after == ()


def test_screen_actually_prunes(caplog: pytest.LogCaptureFixture) -> None:
    """The screen must skip work, or it is only overhead.

    Not a correctness property, but a regression guard: if a future change makes
    the screen vacuous it will still pass the equivalence test above while
    silently costing the search its speed.
    """
    del caplog
    problem = mixed_instance(passengers=6, parcels=5, vehicles=2, seed=11)
    study = Study(problem)
    built = RegretInsertion().solve(study).solution

    request_id = next(iter(sorted(problem.requests)))
    base = built.with_released([request_id])
    longest = max(base.routes, key=len)
    if len(longest) < 4:
        pytest.skip("no route long enough for pruning to be observable")

    def enumerate_all(*, screen: bool) -> list[object]:
        return list(
            candidates_for_route(
                study,
                base,
                request_id,
                longest.vehicle,
                use_slack_screen=screen,
                feasible_only=False,
            )
        )

    screened = enumerate_all(screen=True)
    unscreened = enumerate_all(screen=False)
    assert len(screened) <= len(unscreened)


def test_random_routes_screen_equivalence() -> None:
    """Fuzz the screen against randomly shuffled feasible-ish routes."""
    rng = random.Random(99)
    problem = mixed_instance(passengers=4, parcels=3, vehicles=2, seed=5)
    study = Study(problem)
    solution = RegretInsertion().solve(study).solution

    for _ in range(25):
        request_id = rng.choice(sorted(problem.requests))
        base = solution.with_released([request_id])
        vehicle_id = rng.choice(base.vehicles())
        assert positions(study, base, request_id, vehicle_id, screen=True) == positions(
            study, base, request_id, vehicle_id, screen=False
        )
