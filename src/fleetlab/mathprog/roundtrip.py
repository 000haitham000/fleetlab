"""Moving between a :class:`~fleetlab.domain.solution.Solution` and a variable assignment.

Both directions matter, and for different reasons.

**Solution to assignment** gives a solver a warm start. A good heuristic
solution as an incumbent often matters more to a branch-and-bound run than any
amount of formulation tuning, and it is free -- the study already has one.

**Assignment to solution** is what makes the comparison honest. A solver's
answer comes back as a set of selected arcs; turning it into a
:class:`~fleetlab.domain.solution.Solution` means it can be handed to the *same*
evaluator, the *same* constraints and the *same* objective as every heuristic's
answer. Without this, "the MIP got 412 and the heuristic got 430" compares two
numbers that were computed by different code, and any disagreement between the
formulation and the evaluator stays invisible.

It is also the foundation of matheuristics: solve a model over a subset of
requests, read the answer back, splice it into the incumbent.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from fleetlab.domain.schedule import Schedule
from fleetlab.domain.solution import Solution

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fleetlab.domain.action import Action
    from fleetlab.domain.ids import RequestId, VehicleId
    from fleetlab.linear import Var
    from fleetlab.mathprog.variables import NodeIndex, PickupDeliveryVars
    from fleetlab.study import Study


def solution_to_assignment(
    study: Study,
    variables: PickupDeliveryVars,
    solution: Solution,
) -> dict[str, float]:
    """Render a solution as a variable assignment, for use as a warm start.

    Args:
        study: Supplies the timings that fix the continuous variables.
        variables: The model's variable set.
        solution: The solution to render.

    Returns:
        A mapping from variable name to value, covering every variable in the
        model. Variables the solution does not determine are set to zero.
    """
    nodes = variables.nodes
    assignment: dict[str, float] = {var.name: 0.0 for var in _all_vars(variables)}

    for schedule in solution.routes:
        vehicle = schedule.vehicle
        timing = study.timing(schedule)
        path = [nodes.start[vehicle]]
        for action in schedule.actions:
            path.append(_node_of(nodes, action))
        path.append(nodes.finish[vehicle])

        for tail, head in itertools.pairwise(path):
            arc = variables.arc.get((tail, head, vehicle))
            if arc is not None:
                assignment[arc.name] = 1.0

        for position, action in enumerate(schedule.actions):
            node = _node_of(nodes, action)
            assignment[variables.service_start[node].name] = timing.stops[position].service_start
            for dimension in range(study.problem.arity):
                assignment[variables.load[(node, dimension)].name] = timing.loads[position][
                    dimension
                ]

        assignment[variables.service_start[nodes.start[vehicle]].name] = timing.depot_departure
        assignment[variables.service_start[nodes.finish[vehicle]].name] = timing.return_arrival

        for request_id in schedule.requests():
            onboard = timing.onboard_time(request_id)
            if onboard is not None:
                assignment[variables.onboard[request_id].name] = onboard

    for request_id in study.problem.requests:
        served = request_id not in solution.unassigned
        assignment[variables.served[request_id].name] = 1.0 if served else 0.0

    return assignment


def assignment_to_solution(
    study: Study,
    variables: PickupDeliveryVars,
    values: Mapping[str, float],
    *,
    threshold: float = 0.5,
) -> Solution:
    """Read a solver's answer back into a solution.

    Args:
        study: Supplies the instance.
        variables: The model's variable set.
        values: Variable name to value, as a solver reports it.
        threshold: Binary variables at or above this count as selected.

    Returns:
        A :class:`~fleetlab.domain.solution.Solution` that can be evaluated by
        the same machinery as any heuristic's answer.
    """
    nodes = variables.nodes
    problem = study.problem

    successor: dict[VehicleId, dict[int, int]] = {vehicle_id: {} for vehicle_id in problem.vehicles}
    for (tail, head, vehicle_id), arc in variables.arc.items():
        if values.get(arc.name, 0.0) >= threshold:
            successor[vehicle_id][tail] = head

    node_action: dict[int, Action] = {}
    for request_id in problem.requests:
        pickup, dropoff = problem.actions_for(request_id)
        node_action[nodes.pickup[request_id]] = pickup
        node_action[nodes.dropoff[request_id]] = dropoff

    routes: list[Schedule] = []
    assigned: set[RequestId] = set()
    for vehicle_id in problem.vehicles:
        actions: list[Action] = []
        node = successor[vehicle_id].get(nodes.start[vehicle_id])
        seen: set[int] = set()
        while node is not None and node != nodes.finish[vehicle_id]:
            if node in seen:
                # A disconnected subtour: the model was solved without adequate
                # elimination, or the values are not a valid incumbent. Stop
                # rather than loop; the caller sees a short route and the
                # feasibility layer will report it.
                break
            seen.add(node)
            action = node_action.get(node)
            if action is not None:
                actions.append(action)
                assigned.add(action.request)
            node = successor[vehicle_id].get(node)
        routes.append(Schedule(vehicle_id, tuple(actions)))

    unassigned = frozenset(problem.requests) - frozenset(assigned)
    return Solution(tuple(routes), unassigned)


def _node_of(nodes: NodeIndex, action: Action) -> int:
    return nodes.pickup[action.request] if action.is_pickup else nodes.dropoff[action.request]


def _all_vars(variables: PickupDeliveryVars) -> tuple[Var, ...]:
    return (
        *variables.arc.values(),
        *variables.service_start,
        *variables.load.values(),
        *variables.onboard.values(),
        *variables.served.values(),
    )
