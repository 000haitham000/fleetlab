"""Coverage: whether every request has to be served.

This is a fleet-level rule, so it is the one place a constraint needs to see the
whole solution rather than one route.

Making it optional is deliberate. Some studies must serve everything -- a
contracted paratransit service, a scheduled distribution round. Others may
decline work, and the interesting question is which requests are worth serving
at what price; there the unassigned pool is part of the answer, not a failure.
Registering or omitting this rule is how a study says which kind it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fleetlab.feasibility.violation import Violation
from fleetlab.linear import equals

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fleetlab.domain.solution import Solution
    from fleetlab.linear import Model
    from fleetlab.mathprog.variables import PickupDeliveryVars
    from fleetlab.timing.context import EvalContext
    from fleetlab.timing.timing import SolutionTiming


@dataclass(frozen=True, slots=True)
class AllRequestsServed:
    """Every request released by now must be on some route.

    Attributes:
        only_released: When true, a request whose ``released_at`` is still in
            the future is not expected to be assigned. This is what makes the
            rule usable mid-simulation, where later requests are not yet known
            to the operator.
    """

    only_released: bool = True

    @property
    def name(self) -> str:
        """Identifier used in reports and penalty weights."""
        return "coverage"

    def check_solution(
        self,
        solution: Solution,
        timings: SolutionTiming,
        ctx: EvalContext,
    ) -> Iterator[Violation]:
        """Report each request that should be served but is not."""
        del timings
        now = min(
            (start.ready_at for start in ctx.starts.values()),
            default=0.0,
        )
        assigned = solution.assigned_requests()
        for request_id in ctx.problem.requests:
            if request_id in assigned:
                continue
            if self.only_released and ctx.problem.request(request_id).released_at > now:
                continue
            yield Violation(
                constraint=self.name,
                magnitude=1.0,
                unit="requests",
                request=request_id,
                detail="released but not assigned to any vehicle",
            )

    def to_model(
        self,
        model: Model,
        variables: PickupDeliveryVars,
        ctx: EvalContext,
    ) -> None:
        """Pin every request's ``served`` variable to one."""
        for request_id in ctx.problem.requests:
            model.add(
                equals(variables.served[request_id], 1.0),
                f"must_serve_{request_id}",
            )
