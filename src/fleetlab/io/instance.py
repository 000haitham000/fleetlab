"""Reading and writing instances as JSON.

A study's instances outlive any one run and are shared between people, so the
format is plain, explicit JSON rather than a pickle: readable in a diff,
editable by hand, and not tied to this package's class layout.

Wall-clock times are converted at this boundary and nowhere else. Inside the
framework everything is minutes since the instance epoch -- see
:mod:`fleetlab.domain.units` for why.

Only :class:`~fleetlab.od.constant.ConstantODMatrix` and
:class:`~fleetlab.od.euclidean.EuclideanODMatrix` are serialised. A measured
time-dependent matrix belongs in its own file next to the instance, not inlined
into it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fleetlab.domain.capacity import Capacity, CapacitySpace
from fleetlab.domain.ids import LoadableId, RequestId, StopId, VehicleId
from fleetlab.domain.loadable import Loadable
from fleetlab.domain.problem import Problem
from fleetlab.domain.request import Request
from fleetlab.domain.stop import Stop
from fleetlab.domain.vehicle import DriverBreak, Vehicle
from fleetlab.domain.window import TimeWindow
from fleetlab.od.constant import ConstantODMatrix
from fleetlab.od.euclidean import EuclideanODMatrix

if TYPE_CHECKING:
    from fleetlab.od.base import ODMatrix

FORMAT_VERSION = 1
"""Bumped whenever the on-disk shape changes incompatibly."""


def to_dict(problem: Problem) -> dict[str, Any]:
    """Render an instance as plain JSON-compatible data."""
    return {
        "format_version": FORMAT_VERSION,
        "name": problem.name,
        "horizon_end": problem.horizon_end,
        "capacity_dimensions": list(problem.capacity_space.dimensions),
        "stops": [
            {"id": stop.id, "x": stop.x, "y": stop.y, "name": stop.name}
            for stop in problem.stops.values()
        ],
        "requests": [_request_to_dict(request) for request in problem.requests.values()],
        "vehicles": [_vehicle_to_dict(vehicle) for vehicle in problem.vehicles.values()],
        "od": _od_to_dict(problem.od, problem),
    }


def from_dict(payload: dict[str, Any]) -> Problem:
    """Rebuild an instance from plain data.

    Raises:
        ValueError: If the format version is unsupported or the OD matrix kind
            is not one this module can rebuild.
    """
    version = payload.get("format_version")
    if version != FORMAT_VERSION:
        msg = f"Unsupported instance format version {version!r}; expected {FORMAT_VERSION}."
        raise ValueError(msg)

    space = CapacitySpace(tuple(payload["capacity_dimensions"]))
    arity = len(space)

    stops = [
        Stop(StopId(entry["id"]), entry.get("x"), entry.get("y"), entry.get("name", ""))
        for entry in payload["stops"]
    ]
    requests = [_request_from_dict(entry, arity) for entry in payload["requests"]]
    vehicles = [_vehicle_from_dict(entry, arity) for entry in payload["vehicles"]]
    od = _od_from_dict(payload["od"], stops)

    return Problem.build(
        name=payload["name"],
        capacity_space=space,
        stops=stops,
        requests=requests,
        vehicles=vehicles,
        od=od,
        horizon_end=payload.get("horizon_end", 1.0e7),
    )


def save(problem: Problem, path: str | Path) -> Path:
    """Write an instance to a JSON file."""
    destination = Path(path)
    destination.write_text(json.dumps(to_dict(problem), indent=2), encoding="utf-8")
    return destination


def load(path: str | Path) -> Problem:
    """Read an instance from a JSON file."""
    return from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ----------------------------------------------------------------- internals


def _request_to_dict(request: Request) -> dict[str, Any]:
    return {
        "id": request.id,
        "origin": request.origin,
        "destination": request.destination,
        "loadables": [
            {
                "id": loadable.id,
                "demand": list(loadable.demand.values),
                "kind": loadable.kind,
                "label": loadable.label,
            }
            for loadable in request.loadables
        ],
        "pickup_window": [request.pickup_window.earliest, request.pickup_window.latest],
        "dropoff_window": [request.dropoff_window.earliest, request.dropoff_window.latest],
        "requested_pickup": request.requested_pickup,
        "requested_dropoff": request.requested_dropoff,
        "promised_pickup": request.promised_pickup,
        "promised_dropoff": request.promised_dropoff,
        "pickup_service_duration": request.pickup_service_duration,
        "dropoff_service_duration": request.dropoff_service_duration,
        "max_onboard_time": request.max_onboard_time,
        "revenue": request.revenue,
        "priority": request.priority,
        "released_at": request.released_at,
        "attributes": dict(request.attributes),
    }


def _request_from_dict(entry: dict[str, Any], arity: int) -> Request:
    loadables = tuple(
        Loadable(
            LoadableId(item["id"]),
            _capacity(item["demand"], arity),
            item.get("kind", "generic"),
            item.get("label", ""),
        )
        for item in entry.get("loadables", ())
    )
    return Request(
        id=RequestId(entry["id"]),
        origin=StopId(entry["origin"]),
        destination=StopId(entry["destination"]),
        loadables=loadables,
        pickup_window=TimeWindow(*entry["pickup_window"]),
        dropoff_window=TimeWindow(*entry["dropoff_window"]),
        requested_pickup=entry.get("requested_pickup"),
        requested_dropoff=entry.get("requested_dropoff"),
        promised_pickup=entry.get("promised_pickup"),
        promised_dropoff=entry.get("promised_dropoff"),
        pickup_service_duration=entry.get("pickup_service_duration", 0.0),
        dropoff_service_duration=entry.get("dropoff_service_duration", 0.0),
        max_onboard_time=entry.get("max_onboard_time"),
        revenue=entry.get("revenue", 0.0),
        priority=entry.get("priority", 0.0),
        released_at=entry.get("released_at", 0.0),
        attributes=tuple(sorted(entry.get("attributes", {}).items())),
    )


def _vehicle_to_dict(vehicle: Vehicle) -> dict[str, Any]:
    return {
        "id": vehicle.id,
        "start_stop": vehicle.start_stop,
        "end_stop": vehicle.end_stop,
        "capacity": list(vehicle.capacity.values),
        "available_from": vehicle.available_from,
        "available_until": vehicle.available_until,
        "breaks": [
            {
                "start": rest.start,
                "end": rest.end,
                "stop": rest.stop,
                "flexible_by": rest.flexible_by,
            }
            for rest in vehicle.breaks
        ],
        "fixed_cost": vehicle.fixed_cost,
        "cost_per_minute": vehicle.cost_per_minute,
        "cost_per_distance": vehicle.cost_per_distance,
        "max_shift_duration": vehicle.max_shift_duration,
        "skills": sorted(vehicle.skills),
    }


def _vehicle_from_dict(entry: dict[str, Any], arity: int) -> Vehicle:
    return Vehicle(
        id=VehicleId(entry["id"]),
        start_stop=StopId(entry["start_stop"]),
        end_stop=StopId(entry["end_stop"]),
        capacity=_capacity(entry["capacity"], arity),
        available_from=entry.get("available_from", 0.0),
        available_until=entry.get("available_until", 1.0e7),
        breaks=tuple(
            DriverBreak(
                rest["start"],
                rest["end"],
                StopId(rest["stop"]) if rest.get("stop") else None,
                rest.get("flexible_by", 0.0),
            )
            for rest in entry.get("breaks", ())
        ),
        fixed_cost=entry.get("fixed_cost", 0.0),
        cost_per_minute=entry.get("cost_per_minute", 0.0),
        cost_per_distance=entry.get("cost_per_distance", 0.0),
        max_shift_duration=entry.get("max_shift_duration"),
        skills=frozenset(entry.get("skills", ())),
    )


def _od_to_dict(od: ODMatrix, problem: Problem) -> dict[str, Any]:
    if isinstance(od, EuclideanODMatrix):
        return {"kind": "euclidean", "speed": od.speed}
    if isinstance(od, ConstantODMatrix):
        pairs = [
            {
                "from": origin,
                "to": destination,
                "duration": od.duration(0.0, origin, destination),
                "distance": od.distance(origin, destination),
            }
            for origin in problem.stops
            for destination in problem.stops
            if origin != destination
        ]
        return {"kind": "constant", "pairs": pairs}
    msg = (
        f"Cannot serialise a {type(od).__name__}. Save a measured or time-dependent "
        "matrix in its own file beside the instance."
    )
    raise ValueError(msg)


def _od_from_dict(entry: dict[str, Any], stops: list[Stop]) -> ODMatrix:
    kind = entry.get("kind")
    if kind == "euclidean":
        return EuclideanODMatrix(stops, speed=entry.get("speed", 1.0))
    if kind == "constant":
        durations = {
            (StopId(pair["from"]), StopId(pair["to"])): pair["duration"] for pair in entry["pairs"]
        }
        distances = {
            (StopId(pair["from"]), StopId(pair["to"])): pair["distance"] for pair in entry["pairs"]
        }
        return ConstantODMatrix(durations, distances)
    msg = f"Unknown OD matrix kind {kind!r}."
    raise ValueError(msg)


def _capacity(values: list[float], arity: int) -> Capacity:
    if len(values) != arity:
        msg = f"Capacity vector has {len(values)} entries but the space has {arity}."
        raise ValueError(msg)
    return Capacity(tuple(float(value) for value in values))
