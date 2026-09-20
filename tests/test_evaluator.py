"""The forward pass. Every expected value here is computable by hand.

The layout is five stops on a line one unit apart at unit speed, so a leg of
three units is three minutes and there is nowhere for an arithmetic error to
hide.
"""

from __future__ import annotations

import pytest

from fleetlab.domain import ActionType, Problem, RequestId, Schedule, VehicleId
from fleetlab.domain.vehicle import DriverBreak
from fleetlab.timing import EvalContext, PunctualPolicy, evaluate_route
from fleetlab.timing.policy import EarlyArrivalPolicy


def route(problem: Problem, *pairs: tuple[str, ActionType]) -> Schedule:
    actions = []
    for request_id, kind in pairs:
        pickup, dropoff = problem.actions_for(RequestId(request_id))
        actions.append(pickup if kind is ActionType.PICKUP else dropoff)
    return Schedule(VehicleId("v1"), tuple(actions))


P = ActionType.PICKUP
D = ActionType.DROPOFF


def test_interleaved_route_timings_are_exact(line_problem: Problem) -> None:
    schedule = route(line_problem, ("r1", P), ("r2", P), ("r1", D), ("r2", D))
    timing = evaluate_route(schedule, EvalContext(line_problem))

    # Leaves just in time to reach a (3 units away) for the 20:00 pickup.
    assert timing.depot_departure == pytest.approx(17.0)

    arrivals = [stop.arrival for stop in timing.stops]
    services = [stop.service_start for stop in timing.stops]
    departures = [stop.departure for stop in timing.stops]

    assert arrivals == pytest.approx([20.0, 25.0, 36.0, 41.0])
    # At b the vehicle is 5 min early for r2's 30:00 target, so it holds.
    assert services == pytest.approx([20.0, 30.0, 36.0, 41.0])
    assert departures == pytest.approx([22.0, 33.0, 38.0, 44.0])

    # Return leg from d back to the depot is 12 units.
    assert timing.return_arrival == pytest.approx(56.0)
    assert timing.travel_time == pytest.approx(3.0 + 3.0 + 3.0 + 3.0 + 12.0)


def test_early_at_pickup_waits_but_early_at_dropoff_does_not(line_problem: Problem) -> None:
    schedule = route(line_problem, ("r1", P), ("r2", P), ("r1", D), ("r2", D))
    timing = evaluate_route(schedule, EvalContext(line_problem))

    assert timing.stops[1].wait == pytest.approx(5.0)  # pickup: held to the target
    assert timing.stops[2].wait == pytest.approx(0.0)  # dropoff: served on arrival
    assert timing.total_wait == pytest.approx(5.0)


def test_punctual_policy_also_holds_at_dropoff(line_problem: Problem) -> None:
    schedule = route(line_problem, ("r1", P), ("r1", D))
    early = evaluate_route(schedule, EvalContext(line_problem))
    punctual = evaluate_route(schedule, EvalContext(line_problem, policy=PunctualPolicy()))
    # r1's dropoff window opens at 20:00; the early policy serves on arrival,
    # the punctual policy would too here, but the hold-at-dropoff variant must not.
    held = evaluate_route(
        schedule, EvalContext(line_problem, policy=EarlyArrivalPolicy(hold_at_dropoff=True))
    )
    assert early.stops[1].service_start <= held.stops[1].service_start
    assert (
        punctual.stops[0].service_start
        >= line_problem.request(RequestId("r1")).pickup_window.earliest
    )


def test_onboard_and_excess_onboard(line_problem: Problem) -> None:
    schedule = route(line_problem, ("r1", P), ("r2", P), ("r1", D), ("r2", D))
    timing = evaluate_route(schedule, EvalContext(line_problem))

    # r1 departs a at 22:00 and is served at c at 36:00.
    assert timing.onboard_time(RequestId("r1")) == pytest.approx(14.0)
    # Direct a -> c is 6 units, so the detour costs 8 minutes.
    assert timing.excess_onboard_time(RequestId("r1")) == pytest.approx(8.0)


def test_onboard_time_is_none_for_a_half_placed_request(line_problem: Problem) -> None:
    schedule = route(line_problem, ("r1", P))
    timing = evaluate_route(schedule, EvalContext(line_problem))
    # A half-placed request is a legitimate intermediate state, not an error.
    assert timing.onboard_time(RequestId("r1")) is None
    assert timing.excess_onboard_time(RequestId("r1")) is None


def test_load_profile_tracks_both_capacity_dimensions(line_problem: Problem) -> None:
    schedule = route(line_problem, ("r1", P), ("r2", P), ("r1", D), ("r2", D))
    timing = evaluate_route(schedule, EvalContext(line_problem))

    seats = [load[0] for load in timing.loads]
    kilos = [load[1] for load in timing.loads]
    assert seats == pytest.approx([1.0, 1.0, 0.0, 0.0])
    assert kilos == pytest.approx([0.0, 12.0, 12.0, 0.0])


def test_empty_route_does_not_drive(line_problem: Problem) -> None:
    timing = evaluate_route(Schedule(VehicleId("v1")), EvalContext(line_problem))
    assert timing.is_empty
    assert timing.travel_time == pytest.approx(0.0)
    assert timing.travel_distance == pytest.approx(0.0)
    assert timing.route_duration == pytest.approx(0.0)


def test_evaluation_is_pure(line_problem: Problem) -> None:
    schedule = route(line_problem, ("r1", P), ("r1", D))
    before = (schedule.actions, schedule.committed)
    ctx = EvalContext(line_problem)
    first = evaluate_route(schedule, ctx)
    second = evaluate_route(schedule, ctx)
    assert (schedule.actions, schedule.committed) == before
    assert first == second


def test_driver_break_pushes_service(line_problem: Problem) -> None:
    vehicle = line_problem.vehicles[VehicleId("v1")]
    with_break = Problem.build(
        name="line-break",
        capacity_space=line_problem.capacity_space,
        stops=line_problem.stops.values(),
        requests=line_problem.requests.values(),
        vehicles=[
            type(vehicle)(
                id=vehicle.id,
                start_stop=vehicle.start_stop,
                end_stop=vehicle.end_stop,
                capacity=vehicle.capacity,
                available_from=vehicle.available_from,
                available_until=vehicle.available_until,
                breaks=(DriverBreak(19.0, 27.0),),
            )
        ],
        od=line_problem.od,
        horizon_end=line_problem.horizon_end,
    )
    schedule = route(with_break, ("r1", P), ("r1", D))
    timing = evaluate_route(schedule, EvalContext(with_break))
    # Service would start at 20:00, inside the 19:00-27:00 break, so it is pushed out.
    assert timing.stops[0].service_start == pytest.approx(27.0)


def test_breaks_can_be_ignored_by_policy(line_problem: Problem) -> None:
    schedule = route(line_problem, ("r1", P), ("r1", D))
    ctx = EvalContext(line_problem, policy=EarlyArrivalPolicy(respect_breaks=False))
    timing = evaluate_route(schedule, ctx)
    assert timing.stops[0].service_start == pytest.approx(20.0)


def test_shared_stop_discount_is_off_by_default(line_problem: Problem) -> None:
    """The Java original halved dwell at a shared stop.

    That is an assumption,
    so the default here is no discount and the study opts in.
    """
    default = EarlyArrivalPolicy()
    halved = EarlyArrivalPolicy(shared_stop_factor=0.5)
    request = line_problem.request(RequestId("r1"))
    assert default.dwell(
        request=request, action_type=ActionType.PICKUP, shares_stop_with_previous=True
    ) == pytest.approx(2.0)
    assert halved.dwell(
        request=request, action_type=ActionType.PICKUP, shares_stop_with_previous=True
    ) == pytest.approx(1.0)
