"""Construction by insertion.

Two heuristics, both classic, both here mainly so that the framework has a
working baseline from day one -- a new algorithm needs something to be better
than, and a repair operator needs something to call.

:class:`CheapestInsertion` places whichever request is cheapest to place next.
It is fast and it is myopic: it will happily use up the position that the one
awkward request needed.

:class:`RegretInsertion` places whichever request would suffer most from waiting.
Regret-k scores a request by the gap between its best placement and its k-th
best on a different vehicle, so a request with only one home gets placed before
a request with many. In the literature this is consistently the better
construction, and it is the one to use as a repair operator inside large
neighbourhood search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fleetlab.moves.insert import best_insertion, cheapest_by_vehicle
from fleetlab.search.base import RunRecorder

if TYPE_CHECKING:
    from fleetlab.domain.ids import RequestId
    from fleetlab.domain.solution import Solution
    from fleetlab.search.base import RunResult
    from fleetlab.study import Study


@dataclass(frozen=True, slots=True)
class CheapestInsertion:
    """Repeatedly place the request that costs least to place.

    Attributes:
        use_slack_screen: Whether to prune insertion positions with forward time
            slack. Leave on unless you are measuring the screen's effect.
    """

    use_slack_screen: bool = True

    @property
    def name(self) -> str:
        """Identifier for the results table."""
        return "cheapest_insertion"

    def solve(self, study: Study, initial: Solution | None = None) -> RunResult:
        """Place every unassigned request, cheapest first, until none fits."""
        recorder = RunRecorder()
        solution = initial if initial is not None else study.empty_solution()

        while solution.unassigned:
            best_request: RequestId | None = None
            best_candidate = None
            for request_id in sorted(solution.unassigned):
                candidate = best_insertion(
                    study,
                    solution,
                    request_id,
                    use_slack_screen=self.use_slack_screen,
                )
                recorder.count_evaluations(1)
                if candidate is None:
                    continue
                if best_candidate is None or candidate.delta < best_candidate.delta:
                    best_request, best_candidate = request_id, candidate

            if best_request is None or best_candidate is None:
                break

            solution = best_candidate.move.apply(solution)
            recorder.tick()
            recorder.checkpoint(study, solution)

        return recorder.finish(study, solution, {"unplaced": str(len(solution.unassigned))})


@dataclass(frozen=True, slots=True)
class RegretInsertion:
    """Place the request that would lose most by being placed later.

    Attributes:
        k: How many distinct vehicles to compare. ``k=2`` is the common choice;
            higher values look further ahead and cost more per iteration.
        use_slack_screen: Whether to prune insertion positions with forward time
            slack.
    """

    k: int = 3
    use_slack_screen: bool = True

    @property
    def name(self) -> str:
        """Identifier for the results table."""
        return f"regret_{self.k}_insertion"

    def solve(self, study: Study, initial: Solution | None = None) -> RunResult:
        """Place every unassigned request, highest regret first, until none fits."""
        recorder = RunRecorder()
        solution = initial if initial is not None else study.empty_solution()

        while solution.unassigned:
            chosen = None
            best_regret = float("-inf")

            for request_id in sorted(solution.unassigned):
                per_vehicle = cheapest_by_vehicle(
                    study,
                    solution,
                    request_id,
                    use_slack_screen=self.use_slack_screen,
                )
                recorder.count_evaluations(1)
                if not per_vehicle:
                    continue

                ordered = sorted(per_vehicle.values(), key=lambda entry: entry.delta)
                best = ordered[0]
                # A request that fits on only one vehicle has nowhere else to
                # go, so its regret is unbounded: place it before anything that
                # still has alternatives.
                if len(ordered) < self.k:
                    regret = float("inf")
                else:
                    regret = sum(
                        alternative.delta - best.delta for alternative in ordered[1 : self.k]
                    )

                if regret > best_regret or (
                    regret == best_regret and chosen is not None and best.delta < chosen.delta
                ):
                    best_regret, chosen = regret, best

            if chosen is None:
                break

            solution = chosen.move.apply(solution)
            recorder.tick()
            recorder.checkpoint(study, solution)

        return recorder.finish(
            study,
            solution,
            {"k": str(self.k), "unplaced": str(len(solution.unassigned))},
        )
