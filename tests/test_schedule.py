"""Schedules and solutions: immutability and the structural invariants.

Includes a direct regression test for the ``Stream.anyMatch`` bug in the Java
original, where removing several requests stopped at the first successful
removal and silently left the rest in place.
"""

from __future__ import annotations

import pytest

from fleetlab.domain import (
    Action,
    ActionType,
    Problem,
    RequestId,
    Schedule,
    Solution,
    VehicleId,
)


def pair(problem: Problem, request: str) -> tuple[Action, Action]:
    return problem.actions_for(RequestId(request))


def test_schedule_is_immutable_under_transformation(line_problem: Problem) -> None:
    p1, d1 = pair(line_problem, "r1")
    original = Schedule(VehicleId("v1"), (p1, d1))
    derived = original.with_inserted(pair(line_problem, "r2")[0], 1)

    assert original.actions == (p1, d1)
    assert len(derived) == 3
    assert derived is not original


def test_schedules_are_hashable_and_value_equal(line_problem: Problem) -> None:
    p1, d1 = pair(line_problem, "r1")
    first = Schedule(VehicleId("v1"), (p1, d1))
    second = Schedule(VehicleId("v1"), (p1, d1))
    assert first == second
    assert hash(first) == hash(second)
    assert len({first, second}) == 1


def test_without_requests_removes_every_named_request(line_problem: Problem) -> None:
    """Regression: the Java original used anyMatch, which short-circuits.

    Removing two requests removed only the first.
    """
    p1, d1 = pair(line_problem, "r1")
    p2, d2 = pair(line_problem, "r2")
    route = Schedule(VehicleId("v1"), (p1, p2, d1, d2))

    stripped = route.without_requests([RequestId("r1"), RequestId("r2")])
    assert stripped.actions == ()
    assert stripped.requests() == frozenset()


def test_without_request_is_idempotent(line_problem: Problem) -> None:
    p1, d1 = pair(line_problem, "r1")
    route = Schedule(VehicleId("v1"), (p1, d1))
    once = route.without_request(RequestId("r1"))
    twice = once.without_request(RequestId("r1"))
    assert once.actions == ()
    assert twice is once


def test_pair_insertion_refuses_inverted_order(line_problem: Problem) -> None:
    p1, d1 = pair(line_problem, "r1")
    route = Schedule(VehicleId("v1"))
    with pytest.raises(ValueError, match="greater than pickup index"):
        route.with_pair_inserted(p1, 1, d1, 0)


def test_insertion_index_out_of_range_is_a_programmer_error(line_problem: Problem) -> None:
    p1, _ = pair(line_problem, "r1")
    with pytest.raises(IndexError):
        Schedule(VehicleId("v1")).with_inserted(p1, 5)


def test_committed_prefix_bounds_mutable_positions(line_problem: Problem) -> None:
    p1, d1 = pair(line_problem, "r1")
    p2, d2 = pair(line_problem, "r2")
    route = Schedule(VehicleId("v1"), (p1, p2, d1, d2), committed=2)

    assert list(route.mutable_positions()) == [2, 3, 4]
    assert not route.is_mutable_position(1)
    assert route.is_mutable_position(2)


def test_committed_may_not_exceed_length(line_problem: Problem) -> None:
    p1, d1 = pair(line_problem, "r1")
    with pytest.raises(ValueError, match="out of range"):
        Schedule(VehicleId("v1"), (p1, d1), committed=3)


def test_positions_of_returns_none_when_half_present(line_problem: Problem) -> None:
    p1, _ = pair(line_problem, "r1")
    route = Schedule(VehicleId("v1"), (p1,))
    assert route.positions_of(RequestId("r1")) is None
    assert route.position_of(RequestId("r1"), ActionType.PICKUP) == 0


def test_route_buffer_commits_once(line_problem: Problem) -> None:
    p1, d1 = pair(line_problem, "r1")
    p2, d2 = pair(line_problem, "r2")
    base = Schedule(VehicleId("v1"))

    with base.editing() as buffer:
        buffer.insert_pair(p1, 0, d1, 1)
        buffer.insert_pair(p2, 1, d2, 3)
    built = buffer.commit()

    assert base.is_empty
    assert len(built) == 4
    assert built.requests() == {RequestId("r1"), RequestId("r2")}


def test_route_buffer_reverse_segment_is_the_two_opt_primitive(line_problem: Problem) -> None:
    p1, d1 = pair(line_problem, "r1")
    p2, d2 = pair(line_problem, "r2")
    route = Schedule(VehicleId("v1"), (p1, p2, d1, d2))
    with route.editing() as buffer:
        buffer.reverse_segment(1, 2)
    assert buffer.commit().actions == (p1, d1, p2, d2)


def test_solution_rejects_duplicate_vehicle_routes(line_problem: Problem) -> None:
    del line_problem
    first = Schedule(VehicleId("v1"))
    second = Schedule(VehicleId("v1"))
    with pytest.raises(ValueError, match="more than one route"):
        Solution((first, second))


def test_release_moves_requests_to_the_pool(line_problem: Problem) -> None:
    p1, d1 = pair(line_problem, "r1")
    solution = Solution((Schedule(VehicleId("v1"), (p1, d1)),), frozenset())

    released = solution.with_released([RequestId("r1")])
    assert released.route_for(VehicleId("v1")).is_empty
    assert released.unassigned == {RequestId("r1")}
    # The original is untouched, which is what makes rejection free.
    assert len(solution.route_for(VehicleId("v1"))) == 2


def test_solution_editor_commits_once(line_problem: Problem) -> None:
    p1, d1 = pair(line_problem, "r1")
    solution = Solution((Schedule(VehicleId("v1")),), frozenset({RequestId("r1")}))

    with solution.editing() as editor:
        editor.assign(RequestId("r1"), Schedule(VehicleId("v1"), (p1, d1)))
    built = editor.commit()

    assert solution.unassigned == {RequestId("r1")}
    assert built.unassigned == frozenset()
    assert built.assigned_requests() == {RequestId("r1")}
