"""Ruin operators: choosing what to tear out.

Large neighbourhood search works by removing part of a solution and rebuilding
it better. Which part to remove is the whole art, and the three classic answers
are all here.

Every operator returns a :class:`~fleetlab.moves.base.ReleaseRequests` move
rather than a modified solution, so the choice and the change stay separable --
a run log can record what an operator *wanted* to do even on an iteration that
was rejected.

All of them take an explicit ``random.Random``. Reproducibility is not optional
in a comparison study: a result nobody can re-run is not a result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fleetlab.moves.base import ReleaseRequests

if TYPE_CHECKING:
    import random
    from collections.abc import Sequence

    from fleetlab.domain.ids import RequestId, VehicleId
    from fleetlab.domain.solution import Solution
    from fleetlab.study import Study


def _vehicles_holding(solution: Solution, requests: Sequence[RequestId]) -> tuple[VehicleId, ...]:
    targets = frozenset(requests)
    return tuple(schedule.vehicle for schedule in solution.routes if schedule.requests() & targets)


def random_removal(
    solution: Solution,
    count: int,
    rng: random.Random,
) -> ReleaseRequests:
    """Remove ``count`` assigned requests uniformly at random.

    The baseline every other operator is measured against. It diversifies well
    and repairs badly, which is exactly the right behaviour for one member of an
    adaptive portfolio.
    """
    assigned = sorted(solution.assigned_requests())
    if not assigned:
        return ReleaseRequests((), ())
    chosen = tuple(rng.sample(assigned, min(count, len(assigned))))
    return ReleaseRequests(chosen, _vehicles_holding(solution, chosen))


def related_removal(
    study: Study,
    solution: Solution,
    count: int,
    rng: random.Random,
    *,
    distance_weight: float = 1.0,
    time_weight: float = 1.0,
    determinism: float = 5.0,
) -> ReleaseRequests:
    """Remove a cluster of mutually similar requests (Shaw removal).

    Removing requests that resemble one another gives the repair step a real
    chance of recombining them differently. Removing unrelated ones usually
    yields the same solution back.

    Relatedness here is distance between origins and destinations plus
    difference in target times. A study whose notion of similarity differs --
    shared handling equipment, a common corridor -- should write its own operator;
    this one is a reasonable default, not a claim about the domain.

    Args:
        study: Supplies the instance geometry.
        solution: The incumbent.
        count: How many requests to remove.
        rng: Source of randomness.
        distance_weight: Weight on spatial proximity.
        time_weight: Weight on target-time proximity.
        determinism: Higher biases selection harder toward the most related
            candidate. The literature's ``p`` parameter.
    """
    from fleetlab.domain.request import ActionType

    assigned = sorted(solution.assigned_requests())
    if not assigned:
        return ReleaseRequests((), ())
    count = min(count, len(assigned))

    problem = study.problem
    seed = rng.choice(assigned)
    removed = [seed]
    remaining = [request for request in assigned if request != seed]

    def relatedness(left: RequestId, right: RequestId) -> float:
        first = problem.request(left)
        second = problem.request(right)
        space = problem.od.distance(first.origin, second.origin) + problem.od.distance(
            first.destination, second.destination
        )
        left_time = first.target_time(ActionType.PICKUP) or first.pickup_window.earliest
        right_time = second.target_time(ActionType.PICKUP) or second.pickup_window.earliest
        return distance_weight * space + time_weight * abs(left_time - right_time)

    while len(removed) < count and remaining:
        pivot = rng.choice(removed)
        remaining.sort(key=lambda candidate: relatedness(pivot, candidate))
        index = int(len(remaining) * (rng.random() ** determinism))
        removed.append(remaining.pop(min(index, len(remaining) - 1)))

    chosen = tuple(removed)
    return ReleaseRequests(chosen, _vehicles_holding(solution, chosen))


def worst_removal(
    study: Study,
    solution: Solution,
    count: int,
    rng: random.Random,
    *,
    determinism: float = 3.0,
) -> ReleaseRequests:
    """Remove the requests whose presence costs the most.

    Cost here is the saving from taking a request out of its route -- its
    marginal contribution. Requests that are expensive where they sit are the
    ones most likely to belong somewhere else.

    Args:
        study: Supplies costs.
        solution: The incumbent.
        count: How many requests to remove.
        rng: Source of randomness, used to avoid removing the same set every time.
        determinism: Higher biases selection harder toward the most expensive.
    """
    savings: list[tuple[float, RequestId]] = []
    for schedule in solution.routes:
        if schedule.is_empty:
            continue
        current = study.route_cost(schedule)
        for request_id in sorted(schedule.requests()):
            without = schedule.without_request(request_id)
            savings.append((current - study.route_cost(without), request_id))

    if not savings:
        return ReleaseRequests((), ())

    savings.sort(key=lambda entry: -entry[0])
    chosen: list[RequestId] = []
    pool = [request_id for _, request_id in savings]
    for _ in range(min(count, len(pool))):
        index = int(len(pool) * (rng.random() ** determinism))
        chosen.append(pool.pop(min(index, len(pool) - 1)))

    picked = tuple(chosen)
    return ReleaseRequests(picked, _vehicles_holding(solution, picked))


def route_removal(
    solution: Solution,
    rng: random.Random,
    *,
    routes: int = 1,
) -> ReleaseRequests:
    """Empty one or more whole routes.

    The operator that actually reduces fleet size. Cheapest-insertion repair
    will happily keep an extra vehicle busy forever; emptying a route outright
    is what gives the search a chance to do without it.
    """
    used = [schedule for schedule in solution.routes if schedule]
    if not used:
        return ReleaseRequests((), ())
    picked = rng.sample(used, min(routes, len(used)))
    requests: list[RequestId] = []
    for schedule in picked:
        requests.extend(sorted(schedule.requests()))
    return ReleaseRequests(tuple(requests), tuple(schedule.vehicle for schedule in picked))
