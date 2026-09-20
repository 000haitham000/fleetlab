"""Shared fixtures.

``line_study`` is deliberately hand-computable: five stops on a line one unit
apart, unit speed, so every travel time is an integer number of minutes and a
failing assertion can be checked with a pencil. Tests that need scale use the
generated instances instead.
"""

from __future__ import annotations

import pytest

from fleetlab.domain import (
    Capacity,
    CapacitySpace,
    Loadable,
    LoadableId,
    Problem,
    Request,
    RequestId,
    Stop,
    StopId,
    TimeWindow,
    Vehicle,
    VehicleId,
)
from fleetlab.od import EuclideanODMatrix
from fleetlab.study import Study

SPACE = CapacitySpace(("seats", "kg"))


@pytest.fixture
def space() -> CapacitySpace:
    """A two-dimensional capacity space: one people dimension, one goods."""
    return SPACE


@pytest.fixture
def line_problem() -> Problem:
    """Five stops on a line, unit speed, two paired requests.

    Layout (x coordinate)::

        depot(0)  a(3)  b(6)  c(9)  d(12)

    ``r1`` is a passenger from a to c wanting pickup at 20.
    ``r2`` is a parcel from b to d wanting pickup at 30.
    """
    stops = [
        Stop(StopId("depot"), 0.0, 0.0, "depot"),
        Stop(StopId("a"), 3.0, 0.0),
        Stop(StopId("b"), 6.0, 0.0),
        Stop(StopId("c"), 9.0, 0.0),
        Stop(StopId("d"), 12.0, 0.0),
    ]
    requests = [
        Request(
            id=RequestId("r1"),
            origin=StopId("a"),
            destination=StopId("c"),
            loadables=(Loadable(LoadableId("l1"), SPACE.of(seats=1), "passenger"),),
            pickup_window=TimeWindow(10.0, 40.0),
            dropoff_window=TimeWindow(20.0, 90.0),
            requested_pickup=20.0,
            pickup_service_duration=2.0,
            dropoff_service_duration=2.0,
            max_onboard_time=60.0,
        ),
        Request(
            id=RequestId("r2"),
            origin=StopId("b"),
            destination=StopId("d"),
            loadables=(Loadable(LoadableId("p1"), SPACE.of(kg=12.0), "parcel"),),
            pickup_window=TimeWindow(15.0, 60.0),
            dropoff_window=TimeWindow(25.0, 120.0),
            requested_pickup=30.0,
            pickup_service_duration=3.0,
            dropoff_service_duration=3.0,
        ),
    ]
    vehicles = [
        Vehicle(
            id=VehicleId("v1"),
            start_stop=StopId("depot"),
            end_stop=StopId("depot"),
            capacity=SPACE.of(seats=4, kg=500.0),
            available_from=0.0,
            available_until=300.0,
        )
    ]
    return Problem.build(
        name="line",
        capacity_space=SPACE,
        stops=stops,
        requests=requests,
        vehicles=vehicles,
        od=EuclideanODMatrix(stops, speed=1.0),
        horizon_end=300.0,
    )


@pytest.fixture
def line_study(line_problem: Problem) -> Study:
    """A study over :func:`line_problem` with the default rules and objective."""
    return Study(line_problem)


@pytest.fixture
def zero(space: CapacitySpace) -> Capacity:
    """The all-zero capacity vector."""
    return space.zero()
