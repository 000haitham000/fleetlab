"""Building the mathematical program.

This module contributes only the **structural** rows -- assignment, flow
conservation, and the arc-time linking that turns a set of chosen arcs into a
consistent schedule. Everything semantic comes from the study's own
:class:`~fleetlab.feasibility.base.ConstraintSet` and
:class:`~fleetlab.objective.base.Objective`, which render themselves. So the
model a study exports is, by construction, the problem the study's heuristics
are solving.

Two things determine whether this produces a model a solver can actually work
with, and both are handled here rather than left to the user.

**Arc pruning.** An arc whose tail cannot be left early enough to reach its head
before that head's window closes can never be used. Dropping such arcs up front
removes binaries by the thousand on realistic instances, and costs nothing in
solution quality because the dropped arcs were infeasible anyway.

**Tight big-M.** The time-linking row needs a constant large enough to switch
off when its arc is unused, and every unit larger than necessary weakens the
linear relaxation. A per-arc value computed from the two windows is usually
orders of magnitude smaller than a global constant, and it is the difference
between a model that solves and one that does not.

Time dependence
---------------
A mathematical program needs travel time as a **constant**. A time-dependent OD
matrix makes it a function of a decision variable, so
:func:`build_model` requires either a constant matrix or an explicit, named
decision about how to freeze one. It will not quietly linearise an arbitrary
profile: a bound computed against a frozen profile is not a valid bound for the
true problem, and that is exactly the kind of error that survives into a paper
unnoticed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fleetlab.domain.request import ActionType
from fleetlab.linear import LinExpr, Model, VarKind, equals
from fleetlab.mathprog.variables import NodeIndex, PickupDeliveryVars
from fleetlab.od.base import TimeDependence

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fleetlab.domain.ids import VehicleId
    from fleetlab.domain.units import Instant
    from fleetlab.linear import Var
    from fleetlab.study import Study


@dataclass(frozen=True, slots=True)
class BuildReport:
    """What was built, and what was left out.

    Attributes:
        model: The assembled program.
        variables: Its variable set, for reading a solution back.
        arcs_considered: Arcs before pruning.
        arcs_kept: Arcs after pruning.
        dropped_constraints: Names of the study's rules that could not be
            rendered. A non-empty tuple means the model is a **relaxation** of
            the study's problem, so its optimum is a bound rather than an answer.
        dropped_objective_terms: Names of objective terms that could not be
            rendered. A non-empty tuple means the model optimises a different
            function from the search lane.
        frozen_profile_at: The instant a time-dependent matrix was frozen at, if
            one was. When set, the model's optimum is **not** a valid bound for
            the true time-dependent problem.
    """

    model: Model
    variables: PickupDeliveryVars
    arcs_considered: int
    arcs_kept: int
    dropped_constraints: tuple[str, ...] = ()
    dropped_objective_terms: tuple[str, ...] = ()
    frozen_profile_at: Instant | None = None

    @property
    def is_exact(self) -> bool:
        """Whether this model represents the study's problem exactly.

        False means the optimum is a bound or an approximation, not the answer
        to the question the heuristics are answering.
        """
        return (
            not self.dropped_constraints
            and not self.dropped_objective_terms
            and self.frozen_profile_at is None
        )

    def describe(self) -> str:
        """A multi-line report for a run log."""
        lines = [
            self.model.summary(),
            f"arcs {self.arcs_kept}/{self.arcs_considered} kept after time-window pruning",
        ]
        if self.dropped_constraints:
            lines.append(
                "RELAXATION -- rules not in the model: " + ", ".join(self.dropped_constraints)
            )
        if self.dropped_objective_terms:
            lines.append(
                "DIFFERENT OBJECTIVE -- terms not in the model: "
                + ", ".join(self.dropped_objective_terms)
            )
        if self.frozen_profile_at is not None:
            lines.append(
                f"APPROXIMATION -- travel times frozen at t={self.frozen_profile_at:g}; "
                "the optimum is not a valid bound for the time-dependent problem"
            )
        if self.is_exact:
            lines.append("exact model of the study's problem")
        return "\n".join(lines)


def build_model(
    study: Study,
    *,
    freeze_profile_at: Instant | None = None,
    prune_arcs: bool = True,
    allow_unserved: bool = True,
) -> BuildReport:
    """Assemble a mathematical program for a study.

    Args:
        study: Supplies the instance, the rules and the objective.
        freeze_profile_at: For a time-dependent matrix, the instant at which to
            read travel times. Required for an ``ARBITRARY`` matrix and refused
            for a constant one, so that the approximation is always a deliberate,
            recorded choice.
        prune_arcs: Whether to drop arcs that no schedule could use.
        allow_unserved: Whether ``served`` variables are free. Setting this false
            has the same effect as registering
            :class:`~fleetlab.feasibility.coverage.AllRequestsServed`.

    Returns:
        A :class:`BuildReport`. Read :attr:`BuildReport.is_exact` before quoting
        the optimum as anything other than a bound.

    Raises:
        ValueError: If the travel matrix is time-dependent and no freeze instant
            was given, or a freeze instant was given for a constant matrix.
    """
    problem = study.problem
    dependence = problem.od.time_dependence

    if dependence is TimeDependence.CONSTANT and freeze_profile_at is not None:
        msg = (
            "freeze_profile_at was given but the travel matrix is constant; "
            "drop the argument rather than implying an approximation that is not "
            "being made."
        )
        raise ValueError(msg)
    if dependence is not TimeDependence.CONSTANT and freeze_profile_at is None:
        msg = (
            f"The travel matrix is {dependence.value}, so travel time depends on a "
            "decision variable and the model would not be linear. Pass "
            "freeze_profile_at=<instant> to build an explicit approximation, or "
            "use a ConstantODMatrix for an exact model."
        )
        raise ValueError(msg)

    reference: Instant = freeze_profile_at if freeze_profile_at is not None else 0.0
    nodes = NodeIndex.build(problem)
    model = Model()

    service = _service_durations(study, nodes)
    signed = _signed_demands(study, nodes)
    travel, distance = _travel_and_distance(study, nodes, reference)
    windows = _node_windows(study, nodes)

    arcs, considered = _arc_set(study, nodes, travel, service, windows, prune=prune_arcs)
    time_big_m = {
        (tail, head): max(
            0.0, windows[tail][1] + service[tail] + travel[(tail, head)] - windows[head][0]
        )
        for tail, head in {(tail, head) for tail, head, _ in arcs}
    }
    load_big_m = _load_big_m(study)

    arc_vars: dict[tuple[int, int, VehicleId], Var] = {
        (tail, head, vehicle): model.add_var(f"x_{tail}_{head}_{vehicle}", kind=VarKind.BINARY)
        for tail, head, vehicle in arcs
    }
    service_vars: list[Var] = [
        model.add_var(f"B_{node}", lower=windows[node][0], upper=windows[node][1])
        for node in range(nodes.count)
    ]
    load_vars = {
        (node, dimension): model.add_var(
            f"q_{node}_{dimension}", lower=0.0, upper=load_big_m[dimension]
        )
        for node in range(nodes.count)
        for dimension in range(problem.arity)
    }
    onboard_vars = {
        request_id: model.add_var(f"L_{request_id}", lower=0.0, upper=problem.horizon_end)
        for request_id in problem.requests
    }
    served_vars = {
        request_id: model.add_var(
            f"y_{request_id}",
            kind=VarKind.BINARY,
            lower=0.0 if allow_unserved else 1.0,
        )
        for request_id in problem.requests
    }

    variables = PickupDeliveryVars(
        nodes=nodes,
        arc=arc_vars,
        service_start=service_vars,
        load=load_vars,
        onboard=onboard_vars,
        served=served_vars,
        travel=travel,
        distance=distance,
        service=service,
        signed_demand=signed,
        time_big_m=time_big_m,
        load_big_m=load_big_m,
        big_m=problem.horizon_end * 2.0,
    )

    _add_flow_rows(model, variables, study)
    _add_time_linking_rows(model, variables)

    study.constraints.apply_to_model(model, variables, study.context)
    model.minimise(study.objective.to_expr(variables, study.context))

    return BuildReport(
        model=model,
        variables=variables,
        arcs_considered=considered,
        arcs_kept=len(arcs),
        dropped_constraints=study.constraints.non_linearisable(),
        dropped_objective_terms=study.objective.non_linear_terms(),
        frozen_profile_at=freeze_profile_at,
    )


# ----------------------------------------------------------------- structural


def _add_flow_rows(model: Model, variables: PickupDeliveryVars, study: Study) -> None:
    """Each vehicle leaves its start once, reaches its end once, and conserves flow."""
    nodes = variables.nodes
    for vehicle_id in study.problem.vehicles:
        start = nodes.start[vehicle_id]
        finish = nodes.finish[vehicle_id]

        model.add(
            equals(LinExpr.sum(variables.out_of(start, vehicle_id)), 1.0),
            f"leave_start_{vehicle_id}",
        )
        model.add(
            equals(LinExpr.sum(variables.into(finish, vehicle_id)), 1.0),
            f"reach_end_{vehicle_id}",
        )
        for node in nodes.request_nodes():
            balance = LinExpr.sum(variables.out_of(node, vehicle_id)) - LinExpr.sum(
                variables.into(node, vehicle_id)
            )
            model.add(equals(balance, 0.0), f"flow_{node}_{vehicle_id}")


def _add_time_linking_rows(model: Model, variables: PickupDeliveryVars) -> None:
    """Choosing an arc forces the schedule to respect its travel time."""
    for (tail, head, vehicle_id), arc in variables.arc.items():
        big_m = variables.time_big_m.get((tail, head), variables.big_m)
        if big_m <= 0.0:
            continue
        model.add(
            variables.service_start[head]
            >= variables.service_start[tail]
            + variables.service[tail]
            + variables.travel[(tail, head)]
            - big_m * (1 - arc),
            f"link_{tail}_{head}_{vehicle_id}",
        )


# ---------------------------------------------------------------- model data


def _service_durations(study: Study, nodes: NodeIndex) -> tuple[float, ...]:
    durations = [0.0] * nodes.count
    for request_id, request in study.problem.requests.items():
        durations[nodes.pickup[request_id]] = request.pickup_service_duration
        durations[nodes.dropoff[request_id]] = request.dropoff_service_duration
    return tuple(durations)


def _signed_demands(study: Study, nodes: NodeIndex) -> tuple[tuple[float, ...], ...]:
    arity = study.problem.arity
    demands: list[tuple[float, ...]] = [(0.0,) * arity] * nodes.count
    for request_id, request in study.problem.requests.items():
        demand = request.demand(arity)
        demands[nodes.pickup[request_id]] = demand.values
        demands[nodes.dropoff[request_id]] = tuple(-value for value in demand.values)
    return tuple(demands)


def _travel_and_distance(
    study: Study,
    nodes: NodeIndex,
    reference: Instant,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    od = study.problem.od
    travel: dict[tuple[int, int], float] = {}
    distance: dict[tuple[int, int], float] = {}
    for tail in range(nodes.count):
        for head in range(nodes.count):
            if tail == head:
                continue
            origin = nodes.stop_of[tail]
            destination = nodes.stop_of[head]
            travel[(tail, head)] = od.duration(reference, origin, destination)
            distance[(tail, head)] = od.distance(origin, destination)
    return travel, distance


def _node_windows(study: Study, nodes: NodeIndex) -> tuple[tuple[float, float], ...]:
    windows: list[tuple[float, float]] = [(0.0, study.problem.horizon_end)] * nodes.count
    for request_id, request in study.problem.requests.items():
        pickup = request.window_for(ActionType.PICKUP)
        dropoff = request.window_for(ActionType.DROPOFF)
        windows[nodes.pickup[request_id]] = (pickup.earliest, pickup.latest)
        windows[nodes.dropoff[request_id]] = (dropoff.earliest, dropoff.latest)
    for vehicle_id, vehicle in study.problem.vehicles.items():
        span = (vehicle.available_from, vehicle.available_until)
        windows[nodes.start[vehicle_id]] = span
        windows[nodes.finish[vehicle_id]] = span
    return tuple(windows)


def _load_big_m(study: Study) -> tuple[float, ...]:
    """Per-dimension bound: the largest capacity any vehicle offers, plus slack."""
    arity = study.problem.arity
    bounds = [0.0] * arity
    for vehicle in study.problem.vehicles.values():
        for dimension in range(arity):
            bounds[dimension] = max(bounds[dimension], vehicle.capacity[dimension])
    for request in study.problem.requests.values():
        demand = request.demand(arity)
        for dimension in range(arity):
            bounds[dimension] = max(bounds[dimension], demand[dimension])
    return tuple(bound if bound > 0.0 else 1.0 for bound in bounds)


def _arc_set(
    study: Study,
    nodes: NodeIndex,
    travel: dict[tuple[int, int], float],
    service: Sequence[float],
    windows: Sequence[tuple[float, float]],
    *,
    prune: bool,
) -> tuple[tuple[tuple[int, int, VehicleId], ...], int]:
    """Every arc a schedule could use, and how many were considered.

    Excluded structurally: self-loops, arcs into a vehicle's start node, arcs out
    of its end node, another vehicle's depots, and the arc from a request's
    dropoff back to its own pickup.

    Excluded by pruning: arcs whose tail cannot be left early enough to reach the
    head before that head's window closes.
    """
    problem = study.problem
    request_nodes = nodes.request_nodes()
    own_pickup = {nodes.dropoff[r]: nodes.pickup[r] for r in problem.requests}

    arcs: list[tuple[int, int, VehicleId]] = []
    considered = 0

    for vehicle_id in problem.vehicles:
        start = nodes.start[vehicle_id]
        finish = nodes.finish[vehicle_id]
        tails = (start, *request_nodes)
        heads = (*request_nodes, finish)
        for tail in tails:
            for head in heads:
                if tail == head:
                    continue
                if own_pickup.get(tail) == head:
                    continue
                considered += 1
                if prune:
                    earliest_arrival = windows[tail][0] + service[tail] + travel[(tail, head)]
                    if earliest_arrival > windows[head][1]:
                        continue
                arcs.append((tail, head, vehicle_id))

    return tuple(arcs), considered
