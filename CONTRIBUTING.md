# Contributing

Read `ARCHITECTURE.md` first — most rules below are the justification for
constraints the code places on you, and they make more sense with the reasoning
behind them.

## Setup

```bash
uv sync --all-extras --dev
uv run pre-commit install
```

Every change must pass:

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
```

CI runs exactly this on 3.11 and 3.12.

---

## The five rules

### 1. Exceptions are for programmer errors only

**Never raise to signal that a schedule is infeasible.** An exception says only
*no*; it never says *by how much*, and the margin is what every penalty-based
metaheuristic steers on. Raising also implies something was mutated and then
unwound, which the design does not permit.

```python
# wrong
if arrival > window.latest:
    raise InconsistentTimingException(...)

# right
if arrival > window.latest:
    yield Violation(
        constraint=self.name,
        magnitude=arrival - window.latest,
        unit="minutes",
        ...
    )
```

Raise `ValueError`, `KeyError` or `IndexError` for a bad index, an unknown id, a
plan that rewrites history — things that indicate a bug in the caller, not a
property of the problem. If you are unsure which you have: could a correct
algorithm ever produce this? If yes, it is a violation.

### 2. Nothing in the core branches on what is being carried

The platform moves people and goods. A `Loadable` has a `kind` label for
reporting; **the core never reads it**. Differences are expressed by:

- which capacity dimensions a loadable consumes;
- which constraints the study registers;
- whether a request states a `max_onboard_time`.

If you find yourself writing `if loadable.kind == "passenger"`, that is the
signal the logic belongs in a capacity dimension or a pluggable constraint.

### 3. Evaluation is pure and total

`evaluate_route` reads and returns; it writes nothing and raises nothing for a
modelling reason. Any schedule, however nonsensical, has a timing — it just has
violations too. Break this and you break parallel evaluation, memoisation, and
every search that traverses infeasible space.

### 4. A new constraint states itself twice

```python
@dataclass(frozen=True, slots=True)
class MinimumHeadway:
    """Two vehicles may not serve one stop within `gap` minutes."""

    gap: float = 5.0

    @property
    def name(self) -> str:
        return "minimum_headway"

    def check_route(self, schedule, timing, ctx):
        """Search lane: the sequence is known, so evaluate it."""
        ...
        yield Violation(self.name, magnitude=..., unit="minutes", ...)

    def to_model(self, model, variables, ctx):
        """MIP lane: the sequence is unknown, so constrain it."""
        model.add(...)
```

If it genuinely cannot be linearised, **omit `to_model`**. Do not approximate it
silently — `ConstraintSet.non_linearisable()` will name it and the exporter will
report the model as a relaxation, which is the honest outcome. The same applies
to objective terms and `to_expr`.

Register it explicitly rather than adding it to `standard_constraints`:

```python
rules = standard_constraints().with_rules(route_rules=[MinimumHeadway()])
study = Study(problem, constraints=rules)
```

### 5. Mutation lives inside a bounded scope or in the simulator

Two places mutation is correct:

- **Builders** — `RouteBuffer`, `SolutionEditor` — used within one function and
  committed to an immutable value before anything else sees them.
- **`simulation.ExecutionState`**, which owns the clock and is never handed to
  an optimiser.

Everywhere else, return a new value.

---

## Adding an algorithm

Implement `name` and `solve`:

```python
@dataclass(frozen=True, slots=True)
class MyHeuristic:
    seed: int = 20260920  # fix it; an unreproducible result is not a result

    @property
    def name(self) -> str:
        return "my_heuristic"

    def solve(self, study, initial=None):
        recorder = RunRecorder()
        solution = initial if initial is not None else study.empty_solution()
        ...
        recorder.tick()
        recorder.count_evaluations(n)
        recorder.checkpoint(study, solution)
        return recorder.finish(study, solution, {"note": "..."})
```

Use `RunRecorder` rather than your own bookkeeping, so two algorithms' reported
numbers mean the same thing. Count **evaluations** as well as time: wall-clock
confounds algorithm quality with implementation effort, and evaluations are the
honest denominator for a comparison.

`search/lns.py` is the worked reference. It exercises every seam — ruin
operators, insertion, delta evaluation, penalised cost, adaptive weighting.

---

## Testing

- **Hand-computable fixtures for the evaluator.** `tests/conftest.py` puts five
  stops on a line one unit apart at unit speed, so every expected time is an
  integer a reviewer can check with a pencil.
- **Property tests for anything that prunes.** Any optimisation that skips work
  needs a test that it skips only work that did not matter.
  `tests/test_slack.py::test_screen_never_discards_a_feasible_insertion` is the
  pattern, and it is the test that caught a real bug during development.
- **Cross-lane agreement for anything touching the model.** If you change a
  constraint or an objective term, `tests/test_mathprog.py` must still pass:
  the solver's objective and the evaluator's cost for the same solution have to
  agree to the last digit.
- **Determinism.** Any randomised algorithm gets a test that two runs with the
  same seed produce the same answer.

Tests that need a solver use `pytest.importorskip("pulp")`. Mark anything slow
with `@pytest.mark.slow`.

---

## Style

- Google-style docstrings; `ruff` enforces the shape.
- Full type annotations; `mypy --strict` is not negotiable, because the
  protocols *are* the design and mypy is what catches a constraint that does not
  actually implement both faces.
- Prefer a frozen `slots=True` dataclass for anything value-like.
- Explain **why** in comments, not what. The code says what.
- British spelling in prose (`minimise`, `linearisable`) for consistency with
  what is already there.

## Commits and pull requests

- Present tense, imperative: `Add minimum headway constraint`.
- One logical change per commit.
- A pull request that changes a modelling assumption — a service policy default,
  a constraint's semantics, an objective term — says so in the description and
  updates `ARCHITECTURE.md`. Those assumptions are the study's results; they
  should not move silently.
