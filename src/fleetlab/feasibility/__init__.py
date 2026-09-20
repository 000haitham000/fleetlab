"""Feasibility: rules that judge a schedule, in both lanes.

Every constraint here can say what it means twice -- once by evaluating a known
sequence, once by emitting rows that constrain an unknown one. See
:mod:`fleetlab.feasibility.base` for why that matters.

Nothing here raises to signal infeasibility. Checks return
:class:`~fleetlab.feasibility.violation.Violation` objects carrying a magnitude,
because a metaheuristic needs to know *how far* over the line a candidate is,
not merely that it is.
"""

from fleetlab.feasibility.availability import VehicleAvailability
from fleetlab.feasibility.base import (
    ConstraintSet,
    Linearisable,
    RouteConstraint,
    SolutionConstraint,
)
from fleetlab.feasibility.capacity import CapacityLimit
from fleetlab.feasibility.coverage import AllRequestsServed
from fleetlab.feasibility.onboard import MaxOnboardTime
from fleetlab.feasibility.structure import Pairing, Precedence
from fleetlab.feasibility.timewindows import TimeWindows
from fleetlab.feasibility.violation import FEASIBLE, Feasibility, Violation


def standard_constraints(
    *,
    require_full_coverage: bool = True,
    max_onboard_time: bool = True,
) -> ConstraintSet:
    """The rule set most pickup-and-delivery studies start from.

    Args:
        require_full_coverage: Whether leaving a request unassigned is a
            violation. Turn it off for studies where declining work is a
            legitimate decision.
        max_onboard_time: Whether to enforce per-request onboard-time caps. Keep
            it on for people-moving studies; requests that state no cap are
            unaffected either way, so it is harmless for goods.

    Returns:
        A :class:`~fleetlab.feasibility.base.ConstraintSet`. Add study-specific
        rules with :meth:`~fleetlab.feasibility.base.ConstraintSet.with_rules`.
    """
    route_rules: list[RouteConstraint] = [
        Pairing(),
        Precedence(),
        CapacityLimit(),
        TimeWindows(),
        VehicleAvailability(),
    ]
    if max_onboard_time:
        route_rules.append(MaxOnboardTime())

    solution_rules: list[SolutionConstraint] = []
    if require_full_coverage:
        solution_rules.append(AllRequestsServed())

    return ConstraintSet(tuple(route_rules), tuple(solution_rules))


__all__ = [
    "FEASIBLE",
    "AllRequestsServed",
    "CapacityLimit",
    "ConstraintSet",
    "Feasibility",
    "Linearisable",
    "MaxOnboardTime",
    "Pairing",
    "Precedence",
    "RouteConstraint",
    "SolutionConstraint",
    "TimeWindows",
    "VehicleAvailability",
    "Violation",
    "standard_constraints",
]
