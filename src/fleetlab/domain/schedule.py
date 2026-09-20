"""Schedules: one vehicle's ordered visit sequence.

A :class:`Schedule` is immutable data. Every structural operation returns a new
schedule and leaves the original untouched.

Why immutable
-------------
The search lane produces enormous numbers of throwaway candidates and must be
able to reject any of them. In a mutable design, "reject" means restoring the
previous state, which you buy either with a deep copy per candidate or with an
inverse operation per move type that you have to write and test. Here rejection
is simply not rebinding a name.

Three further properties fall out of that, and all three are load-bearing:

* **Parallel evaluation.** Candidates share nothing mutable, so a worker pool
  can evaluate them without any copying or locking.
* **Memoisation.** A schedule is hashable, so its evaluation can be cached. In
  ruin-and-recreate the same partial route recurs constantly.
* **Cached slack.** Forward time slack computed for a route stays valid, because
  nothing can mutate the route underneath it. That is what allows an insertion
  feasibility test in O(1) instead of O(n) -- see :mod:`fleetlab.timing.slack`.

The committed prefix
--------------------
``committed`` is the number of leading actions that have already been executed
or locked by the simulator and may no longer be reordered. Holding it as one
integer, rather than as a status on every action, makes "may I insert here?" an
O(1) comparison instead of a walk looking for a started action.

Convenience
-----------
Building a route one action at a time through ``with_inserted`` allocates a
schedule per step, which is noisy to read in construction heuristics. Use
:meth:`Schedule.editing` for that -- a mutable buffer that produces one
immutable schedule at the end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, Self

from fleetlab.domain.action import Action
from fleetlab.domain.ids import RequestId, VehicleId
from fleetlab.domain.request import ActionType

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


@dataclass(frozen=True, slots=True)
class Schedule:
    """An immutable ordered sequence of actions for one vehicle.

    Attributes:
        vehicle: The vehicle this route belongs to.
        actions: The visit order. Index order is service order.
        committed: Number of leading actions frozen by execution or locking.
    """

    vehicle: VehicleId
    actions: tuple[Action, ...] = ()
    committed: int = 0
    _hash: int | None = field(default=None, compare=False, repr=False, hash=False)

    def __post_init__(self) -> None:
        if not 0 <= self.committed <= len(self.actions):
            msg = (
                f"committed={self.committed} is out of range for a schedule of "
                f"{len(self.actions)} actions."
            )
            raise ValueError(msg)

    # ----------------------------------------------------------- sequence API

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self) -> Iterator[Action]:
        return iter(self.actions)

    def __getitem__(self, index: int) -> Action:
        return self.actions[index]

    def __bool__(self) -> bool:
        return bool(self.actions)

    @property
    def is_empty(self) -> bool:
        """Whether the vehicle has nothing scheduled."""
        return not self.actions

    def __hash__(self) -> int:
        cached = self._hash
        if cached is None:
            cached = hash((self.vehicle, self.actions, self.committed))
            object.__setattr__(self, "_hash", cached)
        return cached

    # ------------------------------------------------------------- inspection

    def requests(self) -> frozenset[RequestId]:
        """Every request with at least one action on this route."""
        return frozenset(action.request for action in self.actions)

    def position_of(self, request: RequestId, action_type: ActionType) -> int | None:
        """Index of one end of a request, or ``None`` if it is not on this route."""
        for index, action in enumerate(self.actions):
            if action.request == request and action.type is action_type:
                return index
        return None

    def positions_of(self, request: RequestId) -> tuple[int, int] | None:
        """Indices of a request's pickup and dropoff, or ``None`` if not fully present.

        Returns ``None`` rather than raising when the request is absent or only
        half-present: a partially placed request is a state the search lane
        passes through legitimately, not an error.
        """
        pickup = self.position_of(request, ActionType.PICKUP)
        dropoff = self.position_of(request, ActionType.DROPOFF)
        if pickup is None or dropoff is None:
            return None
        return (pickup, dropoff)

    def index_of(self, action: Action) -> int | None:
        """First index at which ``action`` occurs, or ``None``."""
        try:
            return self.actions.index(action)
        except ValueError:
            return None

    def is_mutable_position(self, index: int) -> bool:
        """Whether an action may be inserted at, or removed from, this index.

        Positions inside the committed prefix are frozen. Move generators
        should consult this rather than proposing a move that a constraint will
        later reject -- it is cheaper and it keeps violation reports meaningful.
        """
        return self.committed <= index <= len(self.actions)

    def mutable_positions(self) -> range:
        """Every index at which an insertion is structurally allowed."""
        return range(self.committed, len(self.actions) + 1)

    # --------------------------------------------------------- transformations

    def with_inserted(self, action: Action, index: int) -> Schedule:
        """A copy with ``action`` inserted at ``index``.

        Raises:
            IndexError: If ``index`` is outside ``0 .. len(self)``. That is a
                programmer error; it is never how an infeasible move is
                reported.
        """
        if not 0 <= index <= len(self.actions):
            msg = f"Insertion index {index} out of range for {len(self.actions)} actions."
            raise IndexError(msg)
        actions = (*self.actions[:index], action, *self.actions[index:])
        return Schedule(self.vehicle, actions, self.committed)

    def with_pair_inserted(
        self,
        pickup: Action,
        pickup_index: int,
        dropoff: Action,
        dropoff_index: int,
    ) -> Schedule:
        """A copy with a request's two actions inserted together.

        ``dropoff_index`` is interpreted in the sequence *after* the pickup has
        been placed, which is the convention insertion heuristics naturally
        generate. The two ends always move together; there is no API for
        placing one without the other.

        Raises:
            IndexError: If either index is out of range.
            ValueError: If the dropoff would land at or before the pickup.
                Precedence is structural here rather than a reported violation,
                because a schedule that violates it is not a meaningful object
                to evaluate.
        """
        if dropoff_index <= pickup_index:
            msg = f"Dropoff index {dropoff_index} must be greater than pickup index {pickup_index}."
            raise ValueError(msg)
        return self.with_inserted(pickup, pickup_index).with_inserted(dropoff, dropoff_index)

    def without_index(self, index: int) -> Schedule:
        """A copy with the action at ``index`` removed.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        if not 0 <= index < len(self.actions):
            msg = f"Removal index {index} out of range for {len(self.actions)} actions."
            raise IndexError(msg)
        actions = self.actions[:index] + self.actions[index + 1 :]
        committed = min(self.committed, len(actions))
        return Schedule(self.vehicle, actions, committed)

    def without_request(self, request: RequestId) -> Schedule:
        """A copy with both of a request's actions removed.

        Removing a request that is not present returns an equal schedule rather
        than signalling anything: idempotent removal is what ruin operators
        want.
        """
        actions = tuple(action for action in self.actions if action.request != request)
        if len(actions) == len(self.actions):
            return self
        committed = min(self.committed, len(actions))
        return Schedule(self.vehicle, actions, committed)

    def without_requests(self, requests: Iterable[RequestId]) -> Schedule:
        """A copy with every named request removed.

        Note:
            This removes *all* of them. An implementation built on a
            short-circuiting "any" would stop at the first successful removal
            and silently leave the rest on the route.
        """
        targets = frozenset(requests)
        if not targets:
            return self
        actions = tuple(action for action in self.actions if action.request not in targets)
        if len(actions) == len(self.actions):
            return self
        committed = min(self.committed, len(actions))
        return Schedule(self.vehicle, actions, committed)

    def with_committed(self, committed: int) -> Schedule:
        """A copy with a different committed-prefix length."""
        return Schedule(self.vehicle, self.actions, committed)

    def with_actions(self, actions: tuple[Action, ...]) -> Schedule:
        """A copy carrying a wholly different action sequence."""
        return Schedule(self.vehicle, actions, min(self.committed, len(actions)))

    def cleared(self) -> Schedule:
        """A copy retaining only the committed prefix."""
        return Schedule(self.vehicle, self.actions[: self.committed], self.committed)

    # ------------------------------------------------------------- convenience

    def editing(self) -> RouteBuffer:
        """A mutable buffer for building a route, committed back to a schedule.

        Use this where a construction heuristic would otherwise rebind a
        schedule once per inserted request::

            with route.editing() as buffer:
                for request in ordered:
                    buffer.insert_pair(pickup, i, dropoff, j)
            route = buffer.commit()
        """
        return RouteBuffer(self)

    def describe(self) -> str:
        """A one-line rendering such as ``v1: P(a) P(b) | D(a) D(b)``.

        The bar marks the end of the committed prefix.
        """
        parts: list[str] = []
        for index, action in enumerate(self.actions):
            if index == self.committed and index:
                parts.append("|")
            parts.append(str(action))
        if self.committed == len(self.actions) and self.committed:
            parts.append("|")
        return f"{self.vehicle}: {' '.join(parts) if parts else '(empty)'}"

    def __str__(self) -> str:
        return self.describe()


class RouteBuffer:
    """A mutable scratch buffer over an immutable :class:`Schedule`.

    This exists purely for ergonomics and for hot inner loops. It is the
    ``StringBuilder`` to ``Schedule``'s ``String``: mutate freely inside a
    bounded scope, then :meth:`commit` once.

    Mutation is confined to this object. Nothing outside can observe a
    half-built route, so none of the guarantees that immutability buys are
    given up.
    """

    __slots__ = ("_actions", "_committed", "_vehicle")

    def __init__(self, schedule: Schedule) -> None:
        self._vehicle = schedule.vehicle
        self._actions: list[Action] = list(schedule.actions)
        self._committed = schedule.committed

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def __len__(self) -> int:
        return len(self._actions)

    def __getitem__(self, index: int) -> Action:
        return self._actions[index]

    def insert(self, action: Action, index: int) -> None:
        """Insert one action at ``index``."""
        if not 0 <= index <= len(self._actions):
            msg = f"Insertion index {index} out of range for {len(self._actions)} actions."
            raise IndexError(msg)
        self._actions.insert(index, action)

    def insert_pair(
        self,
        pickup: Action,
        pickup_index: int,
        dropoff: Action,
        dropoff_index: int,
    ) -> None:
        """Insert a request's two actions, dropoff index read after the pickup lands."""
        if dropoff_index <= pickup_index:
            msg = f"Dropoff index {dropoff_index} must be greater than pickup index {pickup_index}."
            raise ValueError(msg)
        self.insert(pickup, pickup_index)
        self.insert(dropoff, dropoff_index)

    def remove_request(self, request: RequestId) -> bool:
        """Remove both ends of a request. Returns whether anything was removed."""
        before = len(self._actions)
        self._actions = [action for action in self._actions if action.request != request]
        return len(self._actions) != before

    def move(self, from_index: int, to_index: int) -> None:
        """Relocate one action within the route."""
        action = self._actions.pop(from_index)
        self._actions.insert(to_index, action)

    def swap(self, first: int, second: int) -> None:
        """Exchange two positions."""
        self._actions[first], self._actions[second] = (
            self._actions[second],
            self._actions[first],
        )

    def reverse_segment(self, start: int, end: int) -> None:
        """Reverse a closed index range in place. The 2-opt primitive."""
        self._actions[start : end + 1] = reversed(self._actions[start : end + 1])

    def snapshot(self) -> tuple[Action, ...]:
        """The current sequence, without committing."""
        return tuple(self._actions)

    def commit(self) -> Schedule:
        """Freeze the buffer into an immutable schedule."""
        return Schedule(
            self._vehicle,
            tuple(self._actions),
            min(self._committed, len(self._actions)),
        )
