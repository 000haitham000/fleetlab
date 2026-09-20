"""Simulation: the mutable half, and the boundary that protects the pure half."""

from __future__ import annotations

import pytest

from fleetlab.domain import Problem, RequestId, Schedule, Solution, VehicleId
from fleetlab.io.generate import mixed_instance
from fleetlab.search import RegretInsertion
from fleetlab.simulation import (
    ExecutionState,
    HorizonLocking,
    LockOnboard,
    NoLocking,
    Simulator,
)
from fleetlab.simulation.simulator import _extend_to_whole_pairs
from fleetlab.study import Study


def test_clock_cannot_run_backwards(line_study: Study) -> None:
    state = ExecutionState(line_study)
    state.advance_to(50.0)
    with pytest.raises(ValueError, match="backwards"):
        state.advance_to(10.0)


def test_committing_a_plan_that_rewrites_history_is_refused(
    line_study: Study, line_problem: Problem
) -> None:
    """An optimiser bug must not be able to produce a simulation that could.

    not have happened.
    """
    p1, d1 = line_problem.actions_for(RequestId("r1"))
    p2, d2 = line_problem.actions_for(RequestId("r2"))

    state = ExecutionState(line_study)
    state.initial_plan(Solution((Schedule(VehicleId("v1"), (p1, d1, p2, d2)),), frozenset()))
    state.advance_to(60.0)
    assert state.vehicle(VehicleId("v1")).actions_done > 0

    reordered = Solution((Schedule(VehicleId("v1"), (p2, d2, p1, d1)),), frozenset())
    with pytest.raises(ValueError, match="already performed"):
        state.commit(reordered)


def test_committing_a_plan_that_drops_history_is_refused(
    line_study: Study, line_problem: Problem
) -> None:
    p1, d1 = line_problem.actions_for(RequestId("r1"))
    state = ExecutionState(line_study)
    state.initial_plan(Solution((Schedule(VehicleId("v1"), (p1, d1)),), frozenset()))
    state.advance_to(100.0)

    with pytest.raises(ValueError, match="already performed"):
        state.commit(Solution((Schedule(VehicleId("v1")),), frozenset()))


def test_freeze_gives_the_optimiser_the_real_position(
    line_study: Study, line_problem: Problem
) -> None:
    p1, d1 = line_problem.actions_for(RequestId("r1"))
    state = ExecutionState(line_study)
    state.initial_plan(Solution((Schedule(VehicleId("v1"), (p1, d1)),), frozenset()))
    state.advance_to(25.0)

    frozen = state.freeze()[VehicleId("v1")]
    assert frozen.stop == line_problem.request(RequestId("r1")).origin
    assert frozen.committed_timings
    # Already under way, so it cannot retroactively have left later.
    assert not frozen.allow_just_in_time


def test_locking_never_splits_a_pair(line_problem: Problem) -> None:
    """A frozen prefix ending between a pickup and its dropoff would strand the.

    loadable: the pickup is locked in while the dropoff is released for
    re-planning onto another vehicle.
    """
    p1, d1 = line_problem.actions_for(RequestId("r1"))
    p2, d2 = line_problem.actions_for(RequestId("r2"))
    route = Schedule(VehicleId("v1"), (p1, p2, d1, d2))

    # Freezing just the two pickups leaves both loadables stranded.
    assert _extend_to_whole_pairs(route, 2) == 4
    # Freezing nothing needs no extension.
    assert _extend_to_whole_pairs(route, 0) == 0


def test_no_locking_freezes_only_what_happened(line_problem: Problem) -> None:
    study = Study(line_problem)
    p1, d1 = line_problem.actions_for(RequestId("r1"))
    route = Schedule(VehicleId("v1"), (p1, d1))
    timing = study.timing(route)

    assert NoLocking().committed_prefix(route, timing, study.context, 0.0, 0) == 0
    assert NoLocking().committed_prefix(route, timing, study.context, 0.0, 1) == 1


def test_horizon_locking_freezes_an_imminent_action(line_problem: Problem) -> None:
    study = Study(line_problem)
    p1, d1 = line_problem.actions_for(RequestId("r1"))
    route = Schedule(VehicleId("v1"), (p1, d1))
    timing = study.timing(route)

    far = HorizonLocking(window=1.0, fraction=0.0).committed_prefix(
        route, timing, study.context, 0.0, 0
    )
    near = HorizonLocking(window=1.0, fraction=0.0).committed_prefix(
        route, timing, study.context, timing.stops[0].service_start - 0.5, 0
    )
    assert far == 0
    assert near == 1


def test_lock_onboard_holds_a_picked_up_request(line_problem: Problem) -> None:
    study = Study(line_problem)
    p1, d1 = line_problem.actions_for(RequestId("r1"))
    p2, d2 = line_problem.actions_for(RequestId("r2"))
    route = Schedule(VehicleId("v1"), (p1, p2, d1, d2))
    timing = study.timing(route)

    # One action executed: r1 is aboard, so its dropoff at index 2 is frozen too.
    assert LockOnboard().committed_prefix(route, timing, study.context, 0.0, 1) == 3


def test_simulation_runs_and_conserves_requests() -> None:
    problem = mixed_instance(passengers=4, wheelchair_users=1, parcels=3, vehicles=2, seed=3)
    study = Study(problem)
    result = Simulator(cadence=45.0).run(study, RegretInsertion())

    assert result.epochs
    assert result.served <= frozenset(problem.requests)
    assert not (result.served & result.never_served)
    assert 0.0 <= result.service_rate <= 1.0


def test_a_request_is_never_served_twice() -> None:
    problem = mixed_instance(passengers=4, parcels=3, vehicles=2, seed=8)
    study = Study(problem)
    state = ExecutionState(study)
    state.initial_plan(RegretInsertion().solve(study).solution)
    state.advance_to(problem.horizon_end)

    counts: dict[RequestId, int] = {}
    for vehicle_id in problem.vehicles:
        for request_id in state.vehicle(vehicle_id).served:
            counts[request_id] = counts.get(request_id, 0) + 1
    assert all(count == 1 for count in counts.values())


def test_static_algorithm_runs_unchanged_in_the_dynamic_loop() -> None:
    """The boundary's whole purpose: nothing in the algorithm knows about a clock."""
    problem = mixed_instance(passengers=4, parcels=2, vehicles=2, seed=21)
    study = Study(problem)
    algorithm = RegretInsertion()

    static = algorithm.solve(study)
    dynamic = Simulator(cadence=60.0).run(study, algorithm)

    assert static.solution is not None
    assert dynamic.final_plan is not None
