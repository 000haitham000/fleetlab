"""Loadables: the things being moved.

A *loadable* is whatever occupies vehicle capacity between a pickup and a
dropoff. It may be a person, a wheelchair user, a pallet, a parcel, a cool-box.
The core deliberately has **no enum of kinds and no branch on kind**: the
difference between a passenger and a parcel is expressed by which capacity
dimensions the loadable consumes, and by which constraints the study registers.

``kind`` exists for reporting, instance files and plots. If you find yourself
writing ``if loadable.kind == ...`` inside ``fleetlab``, that is the signal that
something belongs in a capacity dimension or a pluggable constraint instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fleetlab.domain.capacity import Capacity
from fleetlab.domain.ids import LoadableId


@dataclass(frozen=True, slots=True)
class Loadable:
    """One indivisible unit occupying capacity for the length of its trip.

    Attributes:
        id: Stable identifier, unique within an instance.
        demand: Capacity consumed while onboard, as a vector over the
            instance's :class:`~fleetlab.domain.capacity.CapacitySpace`.
        kind: Free-form label for reporting only. The core never branches on it.
        label: Human-readable description for logs and plots.
    """

    id: LoadableId
    demand: Capacity
    kind: str = "generic"
    label: str = field(default="", compare=False)

    def __str__(self) -> str:
        return self.label or self.id
