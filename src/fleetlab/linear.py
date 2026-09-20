"""A small linear-expression algebra.

Just enough to build a mathematical program and write it out as an LP file. It
exists so that :mod:`fleetlab.feasibility` constraints can render themselves
into rows **without the framework depending on any solver**.

That independence is the point. A study that only wants a bound can export an
LP and hand it to CBC, HiGHS, Gurobi or CPLEX with nothing installed from PyPI;
a study that wants to solve in-process installs the ``solvers`` extra and uses
:mod:`fleetlab.mathprog.adapters`. Tests can assert on the structure of a model
-- how many binaries, which rows -- without a solver anywhere in CI.

Operator conventions
--------------------
``<=`` and ``>=`` between an expression and a number build a :class:`Row`.
Equality does **not** use ``==``: overriding it would break hashing and
dictionary lookup on expressions, and silently turn an accidental comparison
into a constraint. Use :func:`equals` instead, which is also easier to grep for
in a formulation.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping


class VarKind(enum.Enum):
    """The domain a variable ranges over."""

    CONTINUOUS = "continuous"
    BINARY = "binary"
    INTEGER = "integer"


class Sense(enum.Enum):
    """The relational operator of a constraint row."""

    LE = "<="
    GE = ">="
    EQ = "="


class Direction(enum.Enum):
    """Whether the objective is minimised or maximised."""

    MINIMISE = "minimise"
    MAXIMISE = "maximise"


@dataclass(frozen=True, slots=True)
class Var:
    """One decision variable.

    Variables are created by :meth:`Model.add_var` and identified by their
    integer index, which is what expressions carry. The name is for LP output
    and diagnostics only.

    Attributes:
        index: Position in the owning model's variable list.
        name: LP-safe identifier.
        kind: The domain it ranges over.
        lower: Lower bound.
        upper: Upper bound.
    """

    index: int
    name: str
    kind: VarKind
    lower: float
    upper: float

    def __add__(self, other: Var | LinExpr | float) -> LinExpr:
        return LinExpr.of(self) + other

    __radd__ = __add__

    def __sub__(self, other: Var | LinExpr | float) -> LinExpr:
        return LinExpr.of(self) - other

    def __rsub__(self, other: Var | LinExpr | float) -> LinExpr:
        return -LinExpr.of(self) + other

    def __mul__(self, factor: float) -> LinExpr:
        return LinExpr({self.index: float(factor)}, 0.0)

    __rmul__ = __mul__

    def __neg__(self) -> LinExpr:
        return LinExpr({self.index: -1.0}, 0.0)

    def __le__(self, other: Var | LinExpr | float) -> Row:
        return LinExpr.of(self) <= other

    def __ge__(self, other: Var | LinExpr | float) -> Row:
        return LinExpr.of(self) >= other


@dataclass(frozen=True, slots=True)
class LinExpr:
    """A linear combination of variables plus a constant.

    Attributes:
        terms: Coefficient per variable index. Zero coefficients are dropped.
        constant: The additive constant.
    """

    terms: Mapping[int, float] = field(default_factory=dict)
    constant: float = 0.0

    @classmethod
    def of(cls, value: Var | LinExpr | float) -> LinExpr:
        """Coerce a variable, expression or number into an expression."""
        if isinstance(value, LinExpr):
            return value
        if isinstance(value, Var):
            return cls({value.index: 1.0}, 0.0)
        return cls({}, float(value))

    @classmethod
    def sum(cls, parts: Iterable[Var | LinExpr | float]) -> LinExpr:
        """Sum many parts into one expression.

        Cheaper than repeated ``+`` because it accumulates into a single dict
        rather than allocating an expression per addition -- which matters when
        a formulation builds rows with thousands of terms.
        """
        terms: dict[int, float] = {}
        constant = 0.0
        for part in parts:
            expression = cls.of(part)
            constant += expression.constant
            for index, coefficient in expression.terms.items():
                terms[index] = terms.get(index, 0.0) + coefficient
        return cls({i: c for i, c in terms.items() if c != 0.0}, constant)

    def __add__(self, other: Var | LinExpr | float) -> LinExpr:
        addend = LinExpr.of(other)
        terms = dict(self.terms)
        for index, coefficient in addend.terms.items():
            merged = terms.get(index, 0.0) + coefficient
            if merged == 0.0:
                terms.pop(index, None)
            else:
                terms[index] = merged
        return LinExpr(terms, self.constant + addend.constant)

    __radd__ = __add__

    def __sub__(self, other: Var | LinExpr | float) -> LinExpr:
        return self + (-LinExpr.of(other))

    def __rsub__(self, other: Var | LinExpr | float) -> LinExpr:
        return (-self) + other

    def __mul__(self, factor: float) -> LinExpr:
        scale = float(factor)
        if scale == 0.0:
            return LinExpr({}, 0.0)
        return LinExpr({i: c * scale for i, c in self.terms.items()}, self.constant * scale)

    __rmul__ = __mul__

    def __neg__(self) -> LinExpr:
        return LinExpr({i: -c for i, c in self.terms.items()}, -self.constant)

    def __le__(self, other: Var | LinExpr | float) -> Row:
        return _row(self, Sense.LE, other)

    def __ge__(self, other: Var | LinExpr | float) -> Row:
        return _row(self, Sense.GE, other)

    @property
    def is_constant(self) -> bool:
        """Whether the expression has no variable terms."""
        return not self.terms

    def __len__(self) -> int:
        return len(self.terms)


@dataclass(frozen=True, slots=True)
class Row:
    """One constraint: ``lhs <sense> rhs`` with all variables on the left.

    Attributes:
        lhs: Variable terms.
        sense: The relational operator.
        rhs: The constant right-hand side.
        name: LP-safe row name. Assigned by the model if left blank.
    """

    lhs: LinExpr
    sense: Sense
    rhs: float
    name: str = ""

    def renamed(self, name: str) -> Row:
        """A copy carrying a name."""
        return Row(self.lhs, self.sense, self.rhs, name)


def equals(left: Var | LinExpr | float, right: Var | LinExpr | float) -> Row:
    """Build an equality row. Use this rather than ``==``.

    ``==`` is deliberately not overloaded: doing so breaks hashing and turns an
    accidental comparison into a silent constraint.
    """
    return _row(LinExpr.of(left), Sense.EQ, right)


def _row(lhs: LinExpr, sense: Sense, rhs: Var | LinExpr | float) -> Row:
    right = LinExpr.of(rhs)
    combined = lhs - LinExpr(right.terms, 0.0)
    return Row(LinExpr(combined.terms, 0.0), sense, right.constant - combined.constant)


class Model:
    """A mathematical program: variables, rows and an objective.

    Attributes are built up by a formulation and by constraint objects rendering
    themselves. Nothing here solves anything.
    """

    __slots__ = ("_direction", "_objective", "_rows", "_used_names", "_vars")

    def __init__(self) -> None:
        self._vars: list[Var] = []
        self._rows: list[Row] = []
        self._objective: LinExpr = LinExpr()
        self._direction = Direction.MINIMISE
        self._used_names: set[str] = set()

    # ------------------------------------------------------------- variables

    def add_var(
        self,
        name: str,
        *,
        kind: VarKind = VarKind.CONTINUOUS,
        lower: float = 0.0,
        upper: float = float("inf"),
    ) -> Var:
        """Create a variable.

        Raises:
            ValueError: If the name is already taken. Duplicate names produce
                LP files that solvers silently misread, so this is caught here.
        """
        safe = _lp_safe(name)
        if safe in self._used_names:
            msg = f"Duplicate variable name {safe!r}."
            raise ValueError(msg)
        self._used_names.add(safe)
        if kind is VarKind.BINARY:
            lower, upper = 0.0, 1.0
        variable = Var(len(self._vars), safe, kind, lower, upper)
        self._vars.append(variable)
        return variable

    @property
    def variables(self) -> tuple[Var, ...]:
        """Every variable, in creation order."""
        return tuple(self._vars)

    def variable_count(self, kind: VarKind | None = None) -> int:
        """How many variables there are, optionally of one kind."""
        if kind is None:
            return len(self._vars)
        return sum(1 for variable in self._vars if variable.kind is kind)

    # ------------------------------------------------------------------ rows

    def add(self, row: Row, name: str = "") -> Row:
        """Append a constraint row, naming it if it has no name."""
        final_name = _lp_safe(name) if name else f"c{len(self._rows)}"
        stored = row.renamed(final_name)
        self._rows.append(stored)
        return stored

    def add_all(self, rows: Iterable[Row], prefix: str = "") -> None:
        """Append many rows, numbering them from a common prefix."""
        for offset, row in enumerate(rows):
            self.add(row, f"{prefix}{offset}" if prefix else "")

    @property
    def rows(self) -> tuple[Row, ...]:
        """Every constraint row, in insertion order."""
        return tuple(self._rows)

    @property
    def row_count(self) -> int:
        """How many constraint rows there are."""
        return len(self._rows)

    def rows_named(self, prefix: str) -> tuple[Row, ...]:
        """Rows whose name starts with ``prefix``. Handy in tests."""
        return tuple(row for row in self._rows if row.name.startswith(prefix))

    # ------------------------------------------------------------- objective

    def minimise(self, expression: LinExpr) -> None:
        """Set a minimisation objective."""
        self._objective = expression
        self._direction = Direction.MINIMISE

    def maximise(self, expression: LinExpr) -> None:
        """Set a maximisation objective."""
        self._objective = expression
        self._direction = Direction.MAXIMISE

    @property
    def objective(self) -> LinExpr:
        """The objective expression."""
        return self._objective

    @property
    def direction(self) -> Direction:
        """Whether the objective is minimised or maximised."""
        return self._direction

    # ------------------------------------------------------------------- IO

    def to_lp(self) -> str:
        """Render the model in CPLEX LP format.

        The result is readable by CBC, HiGHS, SCIP, Gurobi and CPLEX, which is
        what lets a study get a bound without installing anything.
        """
        lines: list[str] = []
        header = "Minimize" if self._direction is Direction.MINIMISE else "Maximize"
        lines.append(header)
        lines.append(f" obj: {_format_terms(self._objective, self._vars)}")

        lines.append("Subject To")
        for row in self._rows:
            body = _format_terms(row.lhs, self._vars)
            lines.append(f" {row.name}: {body} {row.sense.value} {_number(row.rhs)}")

        bounded = [
            variable
            for variable in self._vars
            if variable.kind is not VarKind.BINARY
            and (variable.lower != 0.0 or variable.upper != float("inf"))
        ]
        if bounded:
            lines.append("Bounds")
            for variable in bounded:
                low = "-inf" if variable.lower == float("-inf") else _number(variable.lower)
                high = "+inf" if variable.upper == float("inf") else _number(variable.upper)
                lines.append(f" {low} <= {variable.name} <= {high}")

        binaries = [v.name for v in self._vars if v.kind is VarKind.BINARY]
        if binaries:
            lines.append("Binaries")
            lines.extend(_wrap(binaries))

        integers = [v.name for v in self._vars if v.kind is VarKind.INTEGER]
        if integers:
            lines.append("Generals")
            lines.extend(_wrap(integers))

        lines.append("End")
        return "\n".join(lines) + "\n"

    def summary(self) -> str:
        """A one-line size report, for run logs."""
        return (
            f"{len(self._vars)} vars "
            f"({self.variable_count(VarKind.BINARY)} binary, "
            f"{self.variable_count(VarKind.INTEGER)} integer), "
            f"{len(self._rows)} rows"
        )

    def __str__(self) -> str:
        return f"Model({self.summary()})"

    def __iter__(self) -> Iterator[Row]:
        return iter(self._rows)


# --------------------------------------------------------------------- output


def _format_terms(expression: LinExpr, variables: list[Var]) -> str:
    if not expression.terms:
        return "0"
    pieces: list[str] = []
    for index in sorted(expression.terms):
        coefficient = expression.terms[index]
        sign = "-" if coefficient < 0 else "+"
        magnitude = abs(coefficient)
        name = variables[index].name
        body = name if magnitude == 1.0 else f"{_number(magnitude)} {name}"
        pieces.append(f"{sign} {body}" if pieces else (f"-{body}" if sign == "-" else body))
    text = " ".join(pieces)
    if expression.constant:
        text += f" {'+' if expression.constant > 0 else '-'} {_number(abs(expression.constant))}"
    return text


def _number(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.10g}"


def _wrap(names: list[str], per_line: int = 8) -> list[str]:
    return [" " + " ".join(names[i : i + per_line]) for i in range(0, len(names), per_line)]


def _lp_safe(name: str) -> str:
    """Make a name safe for LP format: no spaces, no leading digit, no operators."""
    cleaned = "".join(character if character.isalnum() else "_" for character in name)
    if not cleaned:
        cleaned = "v"
    if cleaned[0].isdigit():
        cleaned = f"v{cleaned}"
    return cleaned
