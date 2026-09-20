"""Domain vocabulary: vector capacity, windows, and the single target-time rule."""

from __future__ import annotations

import pytest

from fleetlab.domain import (
    ActionType,
    Capacity,
    CapacitySpace,
    Loadable,
    LoadableId,
    Problem,
    Request,
    RequestId,
    StopId,
    TimeWindow,
    window_around,
)
from fleetlab.domain.units import Epoch, clock

# ----------------------------------------------------------------- capacity


def test_capacity_is_a_vector_over_named_dimensions() -> None:
    space = CapacitySpace(("seats", "wheelchair", "kg"))
    passenger = space.of(seats=1)
    parcel = space.of(kg=12.0)

    assert (passenger + parcel).values == (1.0, 0.0, 12.0)
    assert space.describe(passenger + parcel) == "seats=1 kg=12"


def test_capacity_fits_within_checks_every_dimension() -> None:
    space = CapacitySpace(("seats", "kg"))
    limit = space.of(seats=4, kg=100.0)

    assert space.of(seats=4, kg=100.0).fits_within(limit)
    assert not space.of(seats=5, kg=1.0).fits_within(limit)
    assert not space.of(seats=1, kg=101.0).fits_within(limit)


def test_overflow_is_measurable_per_dimension() -> None:
    """Capacity violations need a size, not a boolean, for penalty-based search."""
    space = CapacitySpace(("seats", "kg"))
    overflow = space.of(seats=6, kg=120.0).overflow(space.of(seats=4, kg=100.0))
    assert overflow.values == (2.0, 20.0)


def test_mixing_capacity_spaces_is_a_programmer_error() -> None:
    with pytest.raises(ValueError, match="different arity"):
        Capacity((1.0, 2.0)) + Capacity((1.0,))


def test_capacity_space_rejects_duplicates_and_emptiness() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        CapacitySpace(("seats", "seats"))
    with pytest.raises(ValueError, match="at least one dimension"):
        CapacitySpace(())


def test_unknown_dimension_is_a_key_error() -> None:
    space = CapacitySpace(("seats",))
    with pytest.raises(KeyError, match="wheelchair"):
        space.of(wheelchair=1)


# ------------------------------------------------------------------ windows


def test_window_lateness_and_earliness() -> None:
    window = TimeWindow(10.0, 20.0)
    assert window.lateness(25.0) == pytest.approx(5.0)
    assert window.lateness(15.0) == pytest.approx(0.0)
    assert window.earliness(4.0) == pytest.approx(6.0)
    assert window.width == pytest.approx(10.0)


def test_empty_window_is_refused() -> None:
    with pytest.raises(ValueError, match="empty"):
        TimeWindow(30.0, 10.0)


def test_window_intersection_returns_none_rather_than_raising() -> None:
    """No overlap is a modelling outcome, not a bug."""
    assert TimeWindow(0.0, 5.0).intersect(TimeWindow(10.0, 15.0)) is None
    overlap = TimeWindow(0.0, 10.0).intersect(TimeWindow(5.0, 15.0))
    assert overlap == TimeWindow(5.0, 10.0)


def test_window_around_resolves_tolerances_once() -> None:
    window = window_around(100.0, tolerance_early=10.0, tolerance_late=20.0)
    assert window == TimeWindow(90.0, 120.0)


# -------------------------------------------------------------- target time


def make_request(**kwargs: object) -> Request:
    defaults: dict[str, object] = {
        "id": RequestId("r"),
        "origin": StopId("a"),
        "destination": StopId("b"),
        "loadables": (),
        "pickup_window": TimeWindow(0.0, 100.0),
        "dropoff_window": TimeWindow(0.0, 200.0),
    }
    defaults.update(kwargs)
    return Request(**defaults)  # type: ignore[arg-type]


def test_promised_time_beats_requested_time() -> None:
    """The framework has exactly one definition of a target time.

    With more than one definition, the same schedule could be judged feasible
    by one code path and infeasible by another.
    """
    request = make_request(requested_pickup=50.0, promised_pickup=60.0)
    assert request.target_time(ActionType.PICKUP) == pytest.approx(60.0)
    assert request.has_commitment(ActionType.PICKUP)


def test_requested_time_is_used_when_nothing_is_promised() -> None:
    request = make_request(requested_pickup=50.0)
    assert request.target_time(ActionType.PICKUP) == pytest.approx(50.0)
    assert not request.has_commitment(ActionType.PICKUP)


def test_target_time_is_none_when_only_a_window_is_given() -> None:
    assert make_request().target_time(ActionType.PICKUP) is None


def test_signed_demand_is_positive_at_pickup_and_negative_at_dropoff() -> None:
    space = CapacitySpace(("seats",))
    request = make_request(
        loadables=(Loadable(LoadableId("l"), space.of(seats=2)),),
    )
    assert request.signed_demand(ActionType.PICKUP, 1).values == (2.0,)
    assert request.signed_demand(ActionType.DROPOFF, 1).values == (-2.0,)


# ------------------------------------------------------------------ problem


def test_problem_rejects_a_request_pointing_at_an_unknown_stop(
    line_problem: Problem,
) -> None:
    with pytest.raises(ValueError, match="unknown stop"):
        Problem.build(
            name="broken",
            capacity_space=line_problem.capacity_space,
            stops=[line_problem.stop(StopId("depot"))],
            requests=line_problem.requests.values(),
            vehicles=[],
            od=line_problem.od,
        )


def test_problem_rejects_a_capacity_of_the_wrong_arity(line_problem: Problem) -> None:
    vehicle = line_problem.vehicle(next(iter(line_problem.vehicles)))
    wrong = type(vehicle)(
        id=vehicle.id,
        start_stop=vehicle.start_stop,
        end_stop=vehicle.end_stop,
        capacity=Capacity((1.0, 2.0, 3.0)),
    )
    with pytest.raises(ValueError, match="dimensional capacity"):
        Problem.build(
            name="broken",
            capacity_space=line_problem.capacity_space,
            stops=line_problem.stops.values(),
            requests=[],
            vehicles=[wrong],
            od=line_problem.od,
        )


def test_action_pairs_are_cached_and_identical(line_problem: Problem) -> None:
    first = line_problem.actions_for(RequestId("r1"))
    second = line_problem.actions_for(RequestId("r1"))
    assert first is second


def test_unknown_ids_raise_key_errors(line_problem: Problem) -> None:
    with pytest.raises(KeyError, match="Unknown request"):
        line_problem.request(RequestId("nope"))
    with pytest.raises(KeyError, match="Unknown stop"):
        line_problem.stop(StopId("nope"))


# -------------------------------------------------------------------- units


def test_epoch_round_trips() -> None:
    import datetime as dt

    epoch = Epoch(dt.datetime(2026, 9, 20, 6, 0, tzinfo=dt.UTC))
    instant = epoch.to_instant(dt.datetime(2026, 9, 20, 8, 30, tzinfo=dt.UTC))
    assert instant == pytest.approx(150.0)
    assert epoch.to_datetime(instant) == dt.datetime(2026, 9, 20, 8, 30, tzinfo=dt.UTC)


def test_clock_formats_minutes() -> None:
    assert clock(0.0) == "00:00"
    assert clock(90.0) == "01:30"
    assert clock(1439.0) == "23:59"
