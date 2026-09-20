"""Objectives: what the study is trying to minimise, stated once for both lanes."""

from fleetlab.objective.base import (
    CostBreakdown,
    LinearObjectiveTerm,
    Objective,
    ObjectiveTerm,
)
from fleetlab.objective.terms import (
    DriverWait,
    ExcessOnboardTime,
    FleetSize,
    Lateness,
    RouteDuration,
    TravelDistance,
    TravelTime,
    UnservedPenalty,
)


def standard_objective(
    *,
    distance_weight: float = 1.0,
    excess_onboard_weight: float = 0.5,
    fleet_weight: float = 100.0,
    unserved_weight: float = 10_000.0,
) -> Objective:
    """A serviceable default: cheap routes, short detours, few vehicles, nothing dropped.

    The weights encode a hierarchy rather than a measurement -- an unserved
    request dominates a vehicle, a vehicle dominates distance. Studies should
    replace them with figures from their own cost model; this exists so an
    example runs and so a new algorithm has something to be compared against on
    day one.
    """
    return Objective(
        terms=(
            TravelDistance(),
            ExcessOnboardTime(),
            FleetSize(),
            UnservedPenalty(),
        ),
        weights={
            "travel_distance": distance_weight,
            "excess_onboard": excess_onboard_weight,
            "fleet_size": fleet_weight,
            "unserved": unserved_weight,
        },
    )


__all__ = [
    "CostBreakdown",
    "DriverWait",
    "ExcessOnboardTime",
    "FleetSize",
    "Lateness",
    "LinearObjectiveTerm",
    "Objective",
    "ObjectiveTerm",
    "RouteDuration",
    "TravelDistance",
    "TravelTime",
    "UnservedPenalty",
    "standard_objective",
]
