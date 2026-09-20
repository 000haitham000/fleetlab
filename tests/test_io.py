"""Instance serialisation."""

from __future__ import annotations

from pathlib import Path

import pytest

from fleetlab.domain import Problem, RequestId, VehicleId
from fleetlab.io import from_dict, load, mixed_instance, save, to_dict
from fleetlab.od import ConstantODMatrix, EuclideanODMatrix


def assert_equivalent(left: Problem, right: Problem) -> None:
    assert left.name == right.name
    assert left.capacity_space == right.capacity_space
    assert set(left.stops) == set(right.stops)
    assert set(left.requests) == set(right.requests)
    assert set(left.vehicles) == set(right.vehicles)
    for request_id in left.requests:
        assert left.request(request_id) == right.request(request_id)
    for vehicle_id in left.vehicles:
        assert left.vehicle(vehicle_id) == right.vehicle(vehicle_id)


def test_euclidean_instance_round_trips() -> None:
    original = mixed_instance(passengers=3, wheelchair_users=1, parcels=2, vehicles=2, seed=4)
    restored = from_dict(to_dict(original))
    assert_equivalent(original, restored)


def test_travel_times_survive_the_round_trip() -> None:
    original = mixed_instance(passengers=2, parcels=2, vehicles=1, seed=6)
    restored = from_dict(to_dict(original))
    for a in original.stops:
        for b in original.stops:
            assert original.od.duration(0.0, a, b) == pytest.approx(restored.od.duration(0.0, a, b))


def test_constant_matrix_round_trips() -> None:
    base = mixed_instance(passengers=2, parcels=1, vehicles=1, seed=9)
    durations = {
        (a, b): base.od.duration(0.0, a, b) for a in base.stops for b in base.stops if a != b
    }
    distances = {(a, b): base.od.distance(a, b) for a in base.stops for b in base.stops if a != b}
    with_constant = Problem.build(
        name=base.name,
        capacity_space=base.capacity_space,
        stops=base.stops.values(),
        requests=base.requests.values(),
        vehicles=base.vehicles.values(),
        od=ConstantODMatrix(durations, distances),
        horizon_end=base.horizon_end,
    )
    restored = from_dict(to_dict(with_constant))
    assert isinstance(restored.od, ConstantODMatrix)
    assert_equivalent(with_constant, restored)


def test_file_round_trip(tmp_path: Path) -> None:
    original = mixed_instance(passengers=2, parcels=2, vehicles=2, seed=12)
    path = save(original, tmp_path / "instance.json")
    assert path.exists()
    assert_equivalent(original, load(path))


def test_unsupported_format_version_is_refused() -> None:
    payload = to_dict(mixed_instance(passengers=1, parcels=1, vehicles=1))
    payload["format_version"] = 999
    with pytest.raises(ValueError, match="Unsupported instance format version"):
        from_dict(payload)


def test_mixed_instance_really_mixes_people_and_goods() -> None:
    """The framework's central claim, asserted on the fixture that exercises it."""
    problem = mixed_instance(passengers=3, wheelchair_users=2, parcels=4, vehicles=2)
    kinds = {
        loadable.kind for request in problem.requests.values() for loadable in request.loadables
    }
    assert {"passenger", "wheelchair", "parcel"} <= kinds

    # People-carrying requests state an onboard cap; goods do not.
    capped = {
        request.id for request in problem.requests.values() if request.max_onboard_time is not None
    }
    uncapped = {
        request.id for request in problem.requests.values() if request.max_onboard_time is None
    }
    assert capped and uncapped

    # At least one vehicle cannot seat anyone but can carry goods.
    seats = problem.capacity_space.index_of("seats")
    kilos = problem.capacity_space.index_of("kg")
    goods_only = [
        vehicle
        for vehicle in problem.vehicles.values()
        if vehicle.capacity[seats] == 0 and vehicle.capacity[kilos] > 0
    ]
    assert goods_only


def test_euclidean_speed_is_preserved() -> None:
    original = mixed_instance(passengers=1, parcels=1, vehicles=1, speed=0.65, seed=2)
    restored = from_dict(to_dict(original))
    assert isinstance(restored.od, EuclideanODMatrix)
    assert restored.od.speed == pytest.approx(0.65)


def test_vehicle_breaks_survive(tmp_path: Path) -> None:
    original = mixed_instance(passengers=1, parcels=1, vehicles=2, seed=15)
    restored = load(save(original, tmp_path / "breaks.json"))
    for vehicle_id in original.vehicles:
        assert (
            original.vehicle(VehicleId(vehicle_id)).breaks
            == restored.vehicle(VehicleId(vehicle_id)).breaks
        )


def test_request_attributes_survive() -> None:
    original = mixed_instance(passengers=1, parcels=1, vehicles=1, seed=18)
    restored = from_dict(to_dict(original))
    for request_id in original.requests:
        assert (
            original.request(RequestId(request_id)).attributes
            == restored.request(RequestId(request_id)).attributes
        )
