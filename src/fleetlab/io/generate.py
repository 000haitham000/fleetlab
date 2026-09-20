"""Synthetic instance generation.

For tests, examples and scaling experiments. Real studies load measured data;
this exists so that the repository runs out of the box and so that a change to
the evaluator can be regression-tested against instances of known shape.

:func:`mixed_instance` deliberately generates **both** people and goods in one
instance, because that is the case the framework claims to handle and it is the
one most likely to break if someone quietly reintroduces a scalar capacity.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from fleetlab.domain.capacity import CapacitySpace
from fleetlab.domain.ids import LoadableId, RequestId, StopId, VehicleId
from fleetlab.domain.loadable import Loadable
from fleetlab.domain.problem import Problem
from fleetlab.domain.request import Request
from fleetlab.domain.stop import Stop
from fleetlab.domain.units import HOUR
from fleetlab.domain.vehicle import DriverBreak, Vehicle
from fleetlab.domain.window import TimeWindow
from fleetlab.od.euclidean import EuclideanODMatrix

if TYPE_CHECKING:
    from fleetlab.domain.units import Duration, Instant

MIXED_SPACE = CapacitySpace(("seats", "wheelchair", "kg", "volume"))
"""Capacity dimensions covering both people and goods.

A passenger consumes a seat; a wheelchair user consumes a bay; a parcel consumes
mass and volume. A vehicle that cannot carry one of them simply has zero in that
dimension, and the capacity constraint rejects it with no type check anywhere.
"""


def mixed_instance(
    *,
    name: str = "mixed",
    passengers: int = 8,
    wheelchair_users: int = 2,
    parcels: int = 6,
    vehicles: int = 3,
    area: float = 40.0,
    speed: float = 0.8,
    horizon: Duration = 8 * HOUR,
    window_width: Duration = 45.0,
    max_onboard_time: Duration = 60.0,
    seed: int = 20260920,
) -> Problem:
    """Generate an instance carrying people and goods on the same fleet.

    Args:
        name: Instance name.
        passengers: Seated passenger requests.
        wheelchair_users: Wheelchair-bay requests.
        parcels: Goods requests, which state no onboard-time cap.
        vehicles: Fleet size. Vehicles alternate between mixed-capability and
            goods-only, so capacity genuinely binds in different dimensions.
        area: Coordinates are drawn from ``[0, area]`` in both axes.
        speed: Distance units per minute.
        horizon: Length of the operating day.
        window_width: Width of each request's service windows.
        max_onboard_time: Onboard cap applied to people-carrying requests.
        seed: Random seed.

    Returns:
        A :class:`~fleetlab.domain.problem.Problem`.
    """
    rng = random.Random(seed)
    space = MIXED_SPACE

    depot = Stop(StopId("depot"), area / 2, area / 2, "depot")
    stops: list[Stop] = [depot]
    requests: list[Request] = []

    def place(label: str) -> Stop:
        stop = Stop(
            StopId(f"s{len(stops)}"),
            rng.uniform(0.0, area),
            rng.uniform(0.0, area),
            label,
        )
        stops.append(stop)
        return stop

    def make_request(
        index: int,
        kind: str,
        code: str,
        demand_kwargs: dict[str, float],
        *,
        onboard_cap: Duration | None,
        service: Duration,
    ) -> None:
        origin = place(f"{kind}-origin-{index}")
        destination = place(f"{kind}-dest-{index}")
        target: Instant = rng.uniform(30.0, horizon - 150.0)
        direct = ((origin.x or 0) - (destination.x or 0)) ** 2 + (
            (origin.y or 0) - (destination.y or 0)
        ) ** 2
        travel = (direct**0.5) / speed
        requests.append(
            Request(
                id=RequestId(f"{code}{index}"),
                origin=origin.id,
                destination=destination.id,
                loadables=(
                    Loadable(
                        LoadableId(f"{code}{index}-load"),
                        space.of(**demand_kwargs),
                        kind,
                        f"{kind} {index}",
                    ),
                ),
                pickup_window=TimeWindow(target, target + window_width),
                dropoff_window=TimeWindow(
                    target + travel,
                    target + travel + window_width + (onboard_cap or 90.0),
                ),
                requested_pickup=target,
                pickup_service_duration=service,
                dropoff_service_duration=service,
                max_onboard_time=onboard_cap,
                revenue=rng.uniform(8.0, 25.0),
                released_at=0.0,
            )
        )

    for index in range(passengers):
        make_request(
            index, "passenger", "pax", {"seats": 1}, onboard_cap=max_onboard_time, service=1.5
        )
    for index in range(wheelchair_users):
        make_request(
            index,
            "wheelchair",
            "whc",
            {"wheelchair": 1},
            onboard_cap=max_onboard_time,
            service=4.0,
        )
    for index in range(parcels):
        make_request(
            index,
            "parcel",
            "pcl",
            {"kg": rng.uniform(2.0, 40.0), "volume": rng.uniform(0.02, 0.4)},
            onboard_cap=None,
            service=2.0,
        )

    fleet: list[Vehicle] = []
    for index in range(vehicles):
        mixed = index % 2 == 0
        capacity = (
            space.of(seats=6, wheelchair=1, kg=300.0, volume=3.0)
            if mixed
            else space.of(seats=0, wheelchair=0, kg=900.0, volume=9.0)
        )
        midday = horizon / 2
        fleet.append(
            Vehicle(
                id=VehicleId(f"v{index}"),
                start_stop=depot.id,
                end_stop=depot.id,
                capacity=capacity,
                available_from=0.0,
                available_until=horizon,
                breaks=(DriverBreak(midday + index * 20.0, midday + index * 20.0 + 30.0),),
                fixed_cost=120.0 if mixed else 90.0,
                cost_per_distance=1.0,
                max_shift_duration=horizon,
            )
        )

    return Problem.build(
        name=name,
        capacity_space=space,
        stops=stops,
        requests=requests,
        vehicles=fleet,
        od=EuclideanODMatrix(stops, speed=speed),
        horizon_end=horizon,
    )


def tiny_instance(*, name: str = "tiny", seed: int = 7) -> Problem:
    """A four-request instance small enough to solve exactly and reason about by hand.

    Used by the tests that compare the heuristic lane against the mathematical
    programming lane, where the model has to stay small enough to solve in CI.
    """
    return mixed_instance(
        name=name,
        passengers=2,
        wheelchair_users=1,
        parcels=1,
        vehicles=2,
        area=12.0,
        speed=1.0,
        horizon=6 * HOUR,
        seed=seed,
    )
