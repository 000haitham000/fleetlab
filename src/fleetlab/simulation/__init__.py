"""Simulation: the mutable half, kept behind a boundary.

The simulator owns the clock and the fleet's real positions and is freely
mutable, because that is what a timeline is. The optimiser sees only frozen
snapshots, so an algorithm written for the static case runs unchanged here. See
:mod:`fleetlab.simulation.state` for why the two are separated.
"""

from fleetlab.simulation.locking import (
    HorizonLocking,
    LockingPolicy,
    LockOnboard,
    NoLocking,
)
from fleetlab.simulation.simulator import EpochRecord, SimulationResult, Simulator
from fleetlab.simulation.state import ExecutionState, VehicleExecution

__all__ = [
    "EpochRecord",
    "ExecutionState",
    "HorizonLocking",
    "LockOnboard",
    "LockingPolicy",
    "NoLocking",
    "SimulationResult",
    "Simulator",
    "VehicleExecution",
]
