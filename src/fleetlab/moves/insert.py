"""Insertion: placing a request's two actions into a route.

This is the most-executed operation in any pickup-and-delivery heuristic, so it
is where the design's performance claims are cashed in.

The naive scan tries every ``(pickup position, dropoff position)`` pair, builds
the candidate route, times it from scratch and checks it. That is O(n) work per
pair and O(n^3) for a route scan.

Here the pickup position is screened first with forward time slack, in O(1) per
position. A position whose push exceeds the slack cannot work, so its whole
family of dropoff positions is skipped without ever building a schedule. The
screen is conservative by construction -- it never discards a position that could
have worked -- and survivors are confirmed by full evaluation, so the result is
exact regardless of how approximate the screen is.

The screen is only worth applying when it is informative. See
:class:`~fleetlab.timing.slack.ForwardSlack` for when it is exact and when it is
a heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fleetlab.timing.slack import insertion_push

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from fleetlab.domain.action import Action
    from fleetlab.domain.ids import RequestId, VehicleId
    from fleetlab.domain.problem import Problem
    from fleetlab.domain.schedule import Schedule
    from fleetlab.domain.solution import Solution
    from fleetlab.feasibility.violation import Feasibility
    from fleetlab.study import Study


@dataclass(frozen=True, slots=True)
class InsertRequest:
    """Put a request's pickup and dropoff into one vehicle's route.

    The two actions are carried on the move rather than looked up when it is
    applied, so that applying needs nothing but the solution. That is what lets
    a move be logged, replayed, or shipped to a worker process on its own.

    Attributes:
        request: What to insert.
        vehicle: Where to insert it.
        pickup_index: Position for the pickup.
        dropoff_index: Position for the dropoff, read *after* the pickup lands.
        pickup: The pickup action.
        dropoff: The dropoff action.
    """

    request: RequestId
    vehicle: VehicleId
    pickup_index: int
    dropoff_index: int
    pickup: Action
    dropoff: Action

    @classmethod
    def of(
        cls,
        problem: Problem,
        request: RequestId,
        vehicle: VehicleId,
        pickup_index: int,
        dropoff_index: int,
    ) -> InsertRequest:
        """Build a move, taking the request's canonical action pair from the instance."""
        pickup, dropoff = problem.actions_for(request)
        return cls(request, vehicle, pickup_index, dropoff_index, pickup, dropoff)

    @property
    def name(self) -> str:
        """Operator name."""
        return "insert"

    @property
    def touched(self) -> tuple[VehicleId, ...]:
        """Routes this move changes."""
        return (self.vehicle,)

    def build_schedule(self, solution: Solution) -> Schedule:
        """The route this insertion would produce."""
        route = solution.route_for(self.vehicle)
        return route.with_pair_inserted(
            self.pickup, self.pickup_index, self.dropoff, self.dropoff_index
        )

    def apply(self, solution: Solution) -> Solution:
        """The solution with this request placed and removed from the pool."""
        return solution.with_assigned(self.request, self.build_schedule(solution))


@dataclass(frozen=True, slots=True)
class InsertionCandidate:
    """One evaluated way of inserting a request.

    Attributes:
        move: Where it would go.
        schedule: The route that results.
        delta: Change in the route's cost.
        feasibility: What, if anything, it breaks.
    """

    move: InsertRequest
    schedule: Schedule
    delta: float
    feasibility: Feasibility

    @property
    def feasible(self) -> bool:
        """Whether the resulting route satisfies every rule."""
        return self.feasibility.feasible

    def score(self, penalty_weight: float = 1e6) -> float:
        """Cost increase plus weighted violations."""
        return self.delta + penalty_weight * self.feasibility.penalty()

    def __str__(self) -> str:
        where = f"{self.move.vehicle}@{self.move.pickup_index}/{self.move.dropoff_index}"
        mark = "ok" if self.feasible else f"{len(self.feasibility)} violations"
        return f"insert {self.move.request} at {where} delta={self.delta:+.4f} ({mark})"


def candidates_for_route(
    study: Study,
    solution: Solution,
    request: RequestId,
    vehicle: VehicleId,
    *,
    use_slack_screen: bool = True,
    feasible_only: bool = True,
) -> Iterator[InsertionCandidate]:
    """Every way of inserting ``request`` into one vehicle's route.

    Args:
        study: The study to evaluate against.
        solution: The incumbent.
        request: What to insert.
        vehicle: Which route to scan.
        use_slack_screen: Whether to prune pickup positions with forward time
            slack before building candidates.
        feasible_only: Whether to yield only candidates that break nothing.

    Yields:
        One :class:`InsertionCandidate` per surviving position pair.
    """
    route = solution.route_for(vehicle)
    base_cost = study.route_cost(route)
    pickup_action, dropoff_action = study.problem.actions_for(request)

    screen = study.slack(route) if use_slack_screen and route else None
    base_timing = study.timing(route) if screen is not None else None

    for pickup_index in route.mutable_positions():
        if screen is not None and base_timing is not None:
            push = insertion_push(route, base_timing, study.context, pickup_action, pickup_index)
            if not screen.can_absorb(pickup_index - 1, push):
                continue

        with_pickup = route.with_inserted(pickup_action, pickup_index)
        for dropoff_index in range(pickup_index + 1, len(with_pickup) + 1):
            candidate_route = with_pickup.with_inserted(dropoff_action, dropoff_index)
            report = study.check_route(candidate_route)
            if feasible_only and not report.feasible:
                continue
            yield InsertionCandidate(
                move=InsertRequest(
                    request,
                    vehicle,
                    pickup_index,
                    dropoff_index,
                    pickup_action,
                    dropoff_action,
                ),
                schedule=candidate_route,
                delta=study.route_cost(candidate_route) - base_cost,
                feasibility=report,
            )


def candidates(
    study: Study,
    solution: Solution,
    request: RequestId,
    *,
    vehicles: Iterable[VehicleId] | None = None,
    use_slack_screen: bool = True,
    feasible_only: bool = True,
) -> Iterator[InsertionCandidate]:
    """Every way of inserting ``request`` into any vehicle's route."""
    fleet = tuple(vehicles) if vehicles is not None else solution.vehicles()
    for vehicle in fleet:
        yield from candidates_for_route(
            study,
            solution,
            request,
            vehicle,
            use_slack_screen=use_slack_screen,
            feasible_only=feasible_only,
        )


def best_insertion(
    study: Study,
    solution: Solution,
    request: RequestId,
    *,
    vehicles: Iterable[VehicleId] | None = None,
    use_slack_screen: bool = True,
) -> InsertionCandidate | None:
    """The cheapest feasible placement of a request, or ``None`` if there is none.

    Returning ``None`` rather than raising is deliberate: "this request does not
    fit anywhere right now" is an ordinary, expected outcome in a construction
    heuristic, not an error.
    """
    best: InsertionCandidate | None = None
    for candidate in candidates(
        study,
        solution,
        request,
        vehicles=vehicles,
        use_slack_screen=use_slack_screen,
        feasible_only=True,
    ):
        if best is None or candidate.delta < best.delta:
            best = candidate
    return best


def cheapest_by_vehicle(
    study: Study,
    solution: Solution,
    request: RequestId,
    *,
    use_slack_screen: bool = True,
) -> dict[VehicleId, InsertionCandidate]:
    """The cheapest feasible placement per vehicle.

    This is what a regret heuristic needs: regret is the gap between the best
    option and the next best *on a different vehicle*.
    """
    best: dict[VehicleId, InsertionCandidate] = {}
    for candidate in candidates(
        study,
        solution,
        request,
        use_slack_screen=use_slack_screen,
        feasible_only=True,
    ):
        vehicle = candidate.move.vehicle
        current = best.get(vehicle)
        if current is None or candidate.delta < current.delta:
            best[vehicle] = candidate
    return best
