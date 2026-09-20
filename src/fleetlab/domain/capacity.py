"""Multi-dimensional capacity.

The platform moves both people and goods, so capacity cannot be a scalar. A
minibus is limited by seats *and* wheelchair bays; the same vehicle carrying
parcels is limited by payload mass *and* load volume. A design with one integer
"capacity" forces every study that mixes the two into a fudge factor.

So capacity is a **vector over named dimensions**. A :class:`CapacitySpace`
fixes the dimension names and their order once per instance; a :class:`Capacity`
is then just a tuple of floats in that order, which keeps the arithmetic in the
evaluator's inner loop cheap.

Nothing in the core branches on whether a loadable is a person or a parcel. The
difference is expressed entirely by *which dimensions a loadable consumes*::

    space = CapacitySpace(("seats", "wheelchair", "kg", "m3"))
    passenger = space.of(seats=1)
    wheelchair_user = space.of(wheelchair=1)
    parcel = space.of(kg=12.5, m3=0.08)

A vehicle that can carry both simply has a non-zero limit in all four
dimensions. A parcel-only van has ``seats=0``, and the capacity constraint then
rejects passengers for free, with no type check anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


@dataclass(frozen=True, slots=True)
class CapacitySpace:
    """The ordered set of capacity dimensions used by one problem instance.

    Attributes:
        dimensions: Dimension names, in the order used by every
            :class:`Capacity` vector in the instance.
    """

    dimensions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.dimensions:
            msg = "A CapacitySpace needs at least one dimension."
            raise ValueError(msg)
        if len(set(self.dimensions)) != len(self.dimensions):
            msg = f"Duplicate capacity dimensions: {self.dimensions!r}"
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.dimensions)

    def index_of(self, dimension: str) -> int:
        """Return the tuple position of a named dimension.

        Raises:
            KeyError: If the dimension is not part of this space. This is a
                programmer error, not a modelling outcome.
        """
        try:
            return self.dimensions.index(dimension)
        except ValueError:
            msg = f"{dimension!r} is not a dimension of {self.dimensions!r}"
            raise KeyError(msg) from None

    def of(self, **amounts: float) -> Capacity:
        """Build a :class:`Capacity` from keyword amounts; omitted dimensions are zero.

        Raises:
            KeyError: If a keyword does not name a dimension of this space.
        """
        values = [0.0] * len(self.dimensions)
        for name, amount in amounts.items():
            values[self.index_of(name)] = float(amount)
        return Capacity(tuple(values))

    def zero(self) -> Capacity:
        """The all-zero vector in this space."""
        return Capacity((0.0,) * len(self.dimensions))

    def describe(self, capacity: Capacity) -> str:
        """Render a capacity vector as ``seats=2 kg=30`` for logs and reprs."""
        parts = [
            f"{name}={value:g}"
            for name, value in zip(self.dimensions, capacity.values, strict=True)
            if value
        ]
        return " ".join(parts) if parts else "empty"


@dataclass(frozen=True, slots=True)
class Capacity:
    """A vector of capacity amounts, aligned to a :class:`CapacitySpace`.

    The space is not carried on the vector: that would double the memory of
    every load figure in the evaluator, and the space is an instance-level
    constant that is always in scope where it is needed. Mixing vectors from
    different spaces is a programmer error and raises.

    Attributes:
        values: Amounts, one per dimension, in the space's order.
    """

    values: tuple[float, ...]

    def __add__(self, other: Capacity) -> Capacity:
        _require_same_arity(self, other)
        return Capacity(tuple(a + b for a, b in zip(self.values, other.values, strict=True)))

    def __sub__(self, other: Capacity) -> Capacity:
        _require_same_arity(self, other)
        return Capacity(tuple(a - b for a, b in zip(self.values, other.values, strict=True)))

    def __mul__(self, factor: float) -> Capacity:
        return Capacity(tuple(a * factor for a in self.values))

    __rmul__ = __mul__

    def fits_within(self, limit: Capacity) -> bool:
        """Whether every dimension is at or below the corresponding limit."""
        _require_same_arity(self, limit)
        return all(a <= b for a, b in zip(self.values, limit.values, strict=True))

    def overflow(self, limit: Capacity) -> Capacity:
        """Per-dimension amount by which this vector exceeds ``limit``, floored at zero.

        This is what makes capacity violations *measurable* rather than merely
        true or false, which is what penalty-based metaheuristics need.
        """
        _require_same_arity(self, limit)
        return Capacity(
            tuple(max(0.0, a - b) for a, b in zip(self.values, limit.values, strict=True))
        )

    def total(self) -> float:
        """Sum across dimensions. Only meaningful as a crude scalar summary."""
        return sum(self.values)

    def is_zero(self) -> bool:
        """Whether every dimension is zero."""
        return not any(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self) -> Iterator[float]:
        return iter(self.values)

    def __getitem__(self, index: int) -> float:
        return self.values[index]


def _require_same_arity(left: Capacity, right: Capacity) -> None:
    if len(left.values) != len(right.values):
        msg = (
            f"Capacity vectors have different arity ({len(left.values)} vs "
            f"{len(right.values)}); they belong to different CapacitySpaces."
        )
        raise ValueError(msg)


def sum_capacities(capacities: Sequence[Capacity], arity: int) -> Capacity:
    """Sum a sequence of capacity vectors, returning zero of ``arity`` when empty."""
    if not capacities:
        return Capacity((0.0,) * arity)
    total = list(capacities[0].values)
    for capacity in capacities[1:]:
        _require_same_arity(capacities[0], capacity)
        for index, value in enumerate(capacity.values):
            total[index] += value
    return Capacity(tuple(total))
