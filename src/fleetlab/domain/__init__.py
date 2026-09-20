"""Domain entities: the vocabulary the whole framework is written in.

Everything here is immutable value data. Nothing here knows how to evaluate a
schedule, judge it feasible, or improve it -- those live in :mod:`fleetlab.timing`,
:mod:`fleetlab.feasibility` and :mod:`fleetlab.search` respectively.

The vocabulary is deliberately neutral between moving people and moving goods. A
:class:`~fleetlab.domain.loadable.Loadable` is whatever occupies capacity; the
core never branches on whether it is a passenger or a parcel.
"""

from fleetlab.domain.action import Action
from fleetlab.domain.capacity import Capacity, CapacitySpace, sum_capacities
from fleetlab.domain.ids import LoadableId, RequestId, StopId, VehicleId
from fleetlab.domain.loadable import Loadable
from fleetlab.domain.problem import Problem
from fleetlab.domain.request import ActionType, Request
from fleetlab.domain.schedule import RouteBuffer, Schedule
from fleetlab.domain.solution import Solution, SolutionEditor
from fleetlab.domain.stop import Stop
from fleetlab.domain.units import DAY, HOUR, MINUTE, Distance, Duration, Epoch, Instant, clock
from fleetlab.domain.vehicle import DriverBreak, Vehicle
from fleetlab.domain.window import UNBOUNDED, TimeWindow, window_around

__all__ = [
    "DAY",
    "HOUR",
    "MINUTE",
    "UNBOUNDED",
    "Action",
    "ActionType",
    "Capacity",
    "CapacitySpace",
    "Distance",
    "DriverBreak",
    "Duration",
    "Epoch",
    "Instant",
    "Loadable",
    "LoadableId",
    "Problem",
    "Request",
    "RequestId",
    "RouteBuffer",
    "Schedule",
    "Solution",
    "SolutionEditor",
    "Stop",
    "StopId",
    "TimeWindow",
    "Vehicle",
    "VehicleId",
    "clock",
    "sum_capacities",
    "window_around",
]
