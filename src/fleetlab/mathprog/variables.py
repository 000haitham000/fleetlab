"""The variable set of the paired pickup-and-delivery formulation.

Node numbering
--------------
Every request contributes two nodes -- its pickup and its dropoff. Every vehicle
contributes two more -- the start and end of its shift -- which lets the fleet be
heterogeneous and multi-depot without a special case.

That gives a node set of ``2 * requests + 2 * vehicles``. Only request nodes are
shared between vehicles; a vehicle's depot nodes are reachable by that vehicle
alone, which is enforced by simply not creating the arc variables.

Variables
---------
``x[i, j, k]``
    Binary. Vehicle *k* travels directly from node *i* to node *j*.
``B[i]``
    Continuous. When service begins at node *i*. Global rather than per-vehicle,
    because a node is visited by at most one vehicle.
``load[i, d]``
    Continuous. Load in capacity dimension *d* carried immediately after node *i*.
``onboard[r]``
    Continuous. Time request *r* spends aboard, between leaving its origin and
    beginning service at its destination.
``served[r]``
    Binary. Whether request *r* is served at all. Studies that forbid rejection
    pin these to one through
    :class:`~fleetlab.feasibility.coverage.AllRequestsServed`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fleetlab.domain.ids import RequestId, VehicleId

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from fleetlab.domain.ids import StopId
    from fleetlab.domain.problem import Problem
    from fleetlab.linear import Var


@dataclass(frozen=True, slots=True)
class NodeIndex:
    """Maps requests and vehicles onto integer node numbers.

    Attributes:
        pickup: Node number of each request's pickup.
        dropoff: Node number of each request's dropoff.
        start: Node number of each vehicle's shift start.
        finish: Node number of each vehicle's shift end.
        stop_of: Physical stop behind each node.
        request_of: Which request a node belongs to, for request nodes.
        count: Total node count.
    """

    pickup: Mapping[RequestId, int]
    dropoff: Mapping[RequestId, int]
    start: Mapping[VehicleId, int]
    finish: Mapping[VehicleId, int]
    stop_of: Sequence[StopId]
    request_of: Mapping[int, RequestId]
    count: int

    @classmethod
    def build(cls, problem: Problem) -> NodeIndex:
        """Number every node of an instance, requests first then depots."""
        pickup: dict[RequestId, int] = {}
        dropoff: dict[RequestId, int] = {}
        start: dict[VehicleId, int] = {}
        finish: dict[VehicleId, int] = {}
        stops: list[StopId] = []
        request_of: dict[int, RequestId] = {}

        for request_id in problem.requests:
            request = problem.request(request_id)
            pickup[request_id] = len(stops)
            request_of[len(stops)] = request_id
            stops.append(request.origin)
        for request_id in problem.requests:
            request = problem.request(request_id)
            dropoff[request_id] = len(stops)
            request_of[len(stops)] = request_id
            stops.append(request.destination)
        for vehicle_id in problem.vehicles:
            vehicle = problem.vehicle(vehicle_id)
            start[vehicle_id] = len(stops)
            stops.append(vehicle.start_stop)
            finish[vehicle_id] = len(stops)
            stops.append(vehicle.end_stop)

        return cls(
            pickup=pickup,
            dropoff=dropoff,
            start=start,
            finish=finish,
            stop_of=tuple(stops),
            request_of=request_of,
            count=len(stops),
        )

    def request_nodes(self) -> tuple[int, ...]:
        """Every pickup and dropoff node."""
        return tuple(self.pickup.values()) + tuple(self.dropoff.values())

    def nodes_for(self, vehicle: VehicleId) -> tuple[int, ...]:
        """Nodes vehicle ``vehicle`` may visit: every request node plus its own depots."""
        return (*self.request_nodes(), self.start[vehicle], self.finish[vehicle])

    def is_pickup(self, node: int) -> bool:
        """Whether a node is a pickup."""
        return node in self.pickup.values()

    def is_depot(self, node: int) -> bool:
        """Whether a node is a vehicle shift start or end."""
        return node not in self.request_of


@dataclass(frozen=True, slots=True)
class PickupDeliveryVars:
    """Every decision variable of the formulation, indexed for readability.

    Constraint objects receive one of these and read from it. They never create
    variables, which keeps the variable set in one place and makes the model's
    size predictable from the instance alone.

    Attributes:
        nodes: The node numbering.
        arc: ``arc[(i, j, k)]`` binary arc variables.
        service_start: ``service_start[i]`` when service begins at node *i*.
        load: ``load[(i, d)]`` load in dimension *d* after node *i*.
        onboard: ``onboard[r]`` time request *r* spends aboard.
        served: ``served[r]`` whether request *r* is served.
        travel: Linearised travel time between every ordered node pair. This is
            the constant the formulation needs and the reason a time-dependent
            matrix cannot be used directly -- see
            :mod:`fleetlab.mathprog.formulation`.
        distance: Path length between every ordered node pair, for
            distance-based objective terms.
        service: Dwell at each node.
        signed_demand: Load added at each node, per capacity dimension:
            positive at a pickup, negative at a dropoff, zero at a depot.
        time_big_m: Per-arc big-M for the time-propagation rows. Computed per
            arc from the two nodes' windows rather than taken as one global
            constant, because a loose big-M is the single most common reason a
            correct pickup-and-delivery formulation refuses to solve: it
            destroys the linear relaxation and the branch-and-bound tree with
            it.
        load_big_m: Per-dimension big-M for the load-propagation rows.
        big_m: Fallback constant for rows with no tighter bound available.
    """

    nodes: NodeIndex
    arc: Mapping[tuple[int, int, VehicleId], Var]
    service_start: Sequence[Var]
    load: Mapping[tuple[int, int], Var]
    onboard: Mapping[RequestId, Var]
    served: Mapping[RequestId, Var]
    travel: Mapping[tuple[int, int], float]
    distance: Mapping[tuple[int, int], float]
    service: Sequence[float]
    signed_demand: Sequence[tuple[float, ...]]
    time_big_m: Mapping[tuple[int, int], float] = field(default_factory=dict)
    load_big_m: Sequence[float] = ()
    big_m: float = 1.0e6
    _arcs_from: dict[tuple[int, VehicleId], list[int]] = field(
        default_factory=dict, compare=False, repr=False, init=False
    )
    _arcs_into: dict[tuple[int, VehicleId], list[int]] = field(
        default_factory=dict, compare=False, repr=False, init=False
    )

    def __post_init__(self) -> None:
        outgoing: dict[tuple[int, VehicleId], list[int]] = {}
        incoming: dict[tuple[int, VehicleId], list[int]] = {}
        for tail, head, vehicle in self.arc:
            outgoing.setdefault((tail, vehicle), []).append(head)
            incoming.setdefault((head, vehicle), []).append(tail)
        object.__setattr__(self, "_arcs_from", outgoing)
        object.__setattr__(self, "_arcs_into", incoming)

    def out_of(self, node: int, vehicle: VehicleId) -> tuple[Var, ...]:
        """Arc variables leaving ``node`` under ``vehicle``."""
        heads = self._arcs_from.get((node, vehicle), [])
        return tuple(self.arc[(node, head, vehicle)] for head in heads)

    def into(self, node: int, vehicle: VehicleId) -> tuple[Var, ...]:
        """Arc variables entering ``node`` under ``vehicle``."""
        tails = self._arcs_into.get((node, vehicle), [])
        return tuple(self.arc[(tail, node, vehicle)] for tail in tails)

    def visits(self, node: int, vehicle: VehicleId) -> tuple[Var, ...]:
        """Arc variables whose selection means ``vehicle`` visits ``node``."""
        return self.out_of(node, vehicle)
