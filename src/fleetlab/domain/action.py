"""Actions: the atoms a schedule is made of.

An :class:`Action` is a pure value object -- a request id, which end of it is
being served, and where. It carries **no timing and no status**.

That is the central change from the earlier Java design, where an ``Action``
owned its own actual arrival, service and departure times as well as a mutable
status. Because those fields lived on the action, every candidate schedule that
contained the action shared them, so no two candidates could be evaluated
independently and nothing could be compared without first being installed.

Here, timing is produced by the evaluator
(:mod:`fleetlab.timing.evaluator`) and execution history lives in the simulator
(:mod:`fleetlab.simulation.state`). An action is small, hashable, and safe to
share between arbitrarily many candidate schedules.
"""

from __future__ import annotations

from dataclasses import dataclass

from fleetlab.domain.ids import RequestId, StopId
from fleetlab.domain.request import ActionType


@dataclass(frozen=True, slots=True)
class Action:
    """One visit a vehicle makes in service of one end of one request.

    ``stop`` is denormalised from the request for the benefit of the evaluator's
    inner loop, which would otherwise perform a dictionary lookup per leg. This
    is safe precisely because the action is immutable: the copy cannot drift
    from the request it came from.

    Attributes:
        request: The request being served.
        type: Which end of the request this visit serves.
        stop: Where the visit happens.
    """

    request: RequestId
    type: ActionType
    stop: StopId

    @property
    def is_pickup(self) -> bool:
        """Whether this action collects loadables."""
        return self.type is ActionType.PICKUP

    @property
    def is_dropoff(self) -> bool:
        """Whether this action delivers loadables."""
        return self.type is ActionType.DROPOFF

    def counterpart_type(self) -> ActionType:
        """The action type at the other end of the same request."""
        return self.type.other

    def __str__(self) -> str:
        marker = "P" if self.is_pickup else "D"
        return f"{marker}({self.request})"
