# fleetlab — context for AI assistants

Condensed reference for working on this codebase. Dense by design. For a gentle
introduction see `docs/TUTORIAL.md`; for setup see `README.md`.

## What this is

A Python 3.11+ framework for simulation-based optimisation of
pickup-and-delivery problems. Every request is a paired pickup and dropoff
served by one vehicle in that order. Loadables may be people or goods; the core
never distinguishes them. Two solution lanes — heuristic search and mathematical
programming — over one problem statement.

## Invariants — do not break these

1. **Exceptions signal programmer errors only.** Never raise to report an
   infeasible schedule. Checks return `Violation` objects with a `magnitude` in
   natural units. `ValueError` / `KeyError` / `IndexError` are for bad indices,
   unknown ids, plans that rewrite execution history. Test: could a correct
   algorithm produce this state? If yes, it is a violation, not an exception.
2. **No core code branches on `Loadable.kind`.** People/goods differences are
   expressed through capacity dimensions, registered constraints, and whether a
   request sets `max_onboard_time`. `kind` is reporting metadata only.
3. **`evaluate_route` is pure and total.** No mutation, no raising for modelling
   reasons. Every schedule has a timing, however infeasible.
4. **`Schedule` and `Solution` are immutable.** Every structural operation
   returns a new object. Mutation is confined to `RouteBuffer`, `SolutionEditor`
   (bounded scope, commit once) and `simulation.ExecutionState` (owns the clock,
   never handed to an optimiser).
5. **A request's two actions always move together.** No API moves a pickup
   without its dropoff.
6. **Every constraint and objective term states itself in both lanes**, or
   explicitly implements only one and is reported by
   `ConstraintSet.non_linearisable()` / `Objective.non_linear_terms()`.
7. **Never silently linearise a time-dependent OD matrix.** `build_model` raises
   unless given an explicit `freeze_profile_at`, and the resulting `BuildReport`
   records that its optimum is not a valid bound.

## Core types

| type | module | mutability | role |
|---|---|---|---|
| `Stop` | `domain.stop` | frozen | a location, optional x/y |
| `Loadable` | `domain.loadable` | frozen | occupies capacity; `demand: Capacity`, `kind: str` |
| `Request` | `domain.request` | frozen | paired job; windows, service durations, optional `max_onboard_time`, `released_at` |
| `Vehicle` | `domain.vehicle` | frozen | specification only, owns no route |
| `Action` | `domain.action` | frozen | `(request, type, stop)`; no timing, no status |
| `Schedule` | `domain.schedule` | **frozen** | one vehicle's ordered actions + `committed: int` |
| `Solution` | `domain.solution` | **frozen** | `routes: tuple[Schedule,...]` + `unassigned: frozenset` |
| `Problem` | `domain.problem` | frozen, `eq=False` | the instance; caches action pairs |
| `Capacity` / `CapacitySpace` | `domain.capacity` | frozen | vector over named dimensions |
| `Study` | `study` | holds caches | problem + policy + constraints + objective |
| `RouteTiming` | `timing.timing` | frozen | the whole temporal picture of one route |
| `Violation` / `Feasibility` | `feasibility.violation` | frozen | infeasibility as data |
| `Model` / `LinExpr` / `Var` / `Row` | `linear` | `Model` mutable | solver-agnostic LP algebra |
| `ExecutionState` | `simulation.state` | **mutable** | the clock and fleet reality |

## Protocols and their signatures

```python
# feasibility.base
RouteConstraint.check_route(schedule, timing, ctx) -> Iterator[Violation]
SolutionConstraint.check_solution(solution, timings, ctx) -> Iterator[Violation]
Linearisable.to_model(model, variables, ctx) -> None
# all three also require:  @property name -> str

# objective.base
ObjectiveTerm.route_cost(schedule, timing, ctx) -> float     # 0.0 for fleet terms
ObjectiveTerm.fleet_cost(solution, timings, ctx) -> float    # 0.0 for route terms
LinearObjectiveTerm.to_expr(variables, ctx) -> LinExpr

# search.base
Algorithm.solve(study, initial: Solution | None = None) -> RunResult

# moves.base
Move.apply(solution) -> Solution
Move.touched -> tuple[VehicleId, ...]      # drives delta evaluation

# timing.policy
ServicePolicy.service_start(*, request, action_type, arrival, vehicle) -> Instant
ServicePolicy.dwell(*, request, action_type, shares_stop_with_previous) -> Duration

# od.base
ODMatrix.duration(depart_at, origin, destination) -> Duration
ODMatrix.distance(origin, destination) -> Distance
ODMatrix.time_dependence -> TimeDependence   # CONSTANT | PIECEWISE | ARBITRARY

# simulation.locking
LockingPolicy.committed_prefix(schedule, timing, ctx, now, executed) -> int
```

## Module map

| module | contents |
|---|---|
| `domain/` | entities above, plus `units` (Instant/Duration/Epoch), `window` (TimeWindow), `ids` (NewType str ids) |
| `od/` | `base` (protocol + `TimeDependence`), `constant`, `euclidean`, `piecewise` (+ `fifo_violations()`) |
| `timing/` | `evaluator` (the forward pass + memoising `Evaluator`), `timing` (results), `policy` (`EarlyArrivalPolicy`, `PunctualPolicy`), `departure` (just-in-time fixed point), `slack` (forward time slack), `context` (`EvalContext`, `VehicleStart`) |
| `feasibility/` | `base` (protocols + `ConstraintSet`), `violation`, `structure` (Pairing, Precedence), `capacity`, `timewindows`, `availability`, `onboard`, `coverage` |
| `objective/` | `base` (`Objective`, `CostBreakdown`), `terms` (TravelDistance, TravelTime, RouteDuration, DriverWait, FleetSize, ExcessOnboardTime, Lateness, UnservedPenalty) |
| `moves/` | `base` (Move protocol, `delta_cost`, `evaluate_move`), `insert` (candidate enumeration + slack screen), `ruin` (random / related / worst / route removal) |
| `search/` | `base` (`Algorithm`, `RunResult`, `RunRecorder`), `greedy` (Cheapest, Regret-k), `lns` (`AdaptiveLNS`) |
| `mathprog/` | `variables` (`NodeIndex`, `PickupDeliveryVars`), `formulation` (`build_model`, `BuildReport`), `roundtrip` (solution ↔ assignment), `adapters` (optional PuLP, `write_lp`) |
| `simulation/` | `state` (`ExecutionState`, mutable), `locking` (Horizon / None / LockOnboard), `simulator` (decision-epoch loop) |
| `io/` | `instance` (JSON), `generate` (`mixed_instance`, `tiny_instance`, `MIXED_SPACE`) |
| `linear.py` | top-level, no fleetlab deps; LP algebra + `to_lp()` |
| `study.py` | `Study` |

## Design decisions and their reasons

**Timing derived, never stored.** One O(n) forward pass yields arrivals, service
starts, departures, waits, load profile, onboard times, return leg. No cache to
invalidate; a move is a list edit.

**Schedules immutable.** Rejecting a candidate costs nothing. Consequences that
are load-bearing: evaluation is memoisable (keyed on schedule hash); candidates
are parallelisable; forward time slack stays valid, giving O(1) insertion
feasibility instead of O(n).

**Violations carry magnitudes.** Penalty-based metaheuristics steer on the
margin. `Feasibility.penalty(weights, default)` takes per-constraint weights
that can adapt during a run.

**Constraints have two faces.** A mathematical program never asks "given this
order, when do we arrive?" — the order is its output. So the constraint
semantics must exist independently of any sequence. One object states both, so
the two lanes cannot drift.

**Minutes as floats, not datetimes.** A model needs numbers in its rows;
conversion happens only at the IO boundary via `Epoch`. Also ~100× faster in the
innermost loop.

**Absolute time windows, not offsets.** Resolving tolerances at each use site
lets the same window mean different things in different code paths. `TimeWindow`
stores `earliest`/`latest`; `window_around()` builds one from a target plus
tolerances. This is also exactly the `e_i`, `l_i` a model needs.

**One target-time rule.** `Request.target_time(action_type)` = promised if set,
else requested, else `None`. Nothing else may form its own answer.

**`committed` is an integer prefix**, not a per-action status. "May I insert
here?" is `schedule.committed <= index`.

**Service policy injected.** "Wait at a pickup, serve immediately at a dropoff"
is a modelling choice that varies by study and decides whether the model needs
`B_i >= e_i` or `B_i = max(A_i, e_i)`.

**Simulation mutable, search pure.** A timeline is monotone; a search needs
millions of discardable candidates. `ExecutionState.freeze()` is the one-way
boundary. `ExecutionState.commit()` refuses a plan that reorders or drops an
executed action.

## Gotchas

- `ForwardSlack` is a **filter**, not a verdict. Exact only when the matrix is
  `CONSTANT` and no `max_onboard_time` is active (`ForwardSlack.exact`).
  Otherwise optimistic: it may pass a candidate that proves infeasible, never
  rejects one that would have worked. Confirm survivors by full evaluation.
- `insertion_push` at index 0 must re-solve the depot departure via
  `depot_departure_for`. Treating it as fixed wrongly discards feasible
  position-zero insertions, because inserting a new first action lets the
  vehicle leave *earlier*. Regression:
  `tests/test_slack.py::test_screen_never_discards_a_feasible_insertion`.
- `Schedule.with_pair_inserted(pickup, i, dropoff, j)` reads `j` in the sequence
  **after** the pickup has landed.
- `Solution` is `frozen` but builds a `_by_vehicle` index in `__post_init__`;
  `Problem` is `eq=False` (identity-compared, holds dicts).
- `equals(a, b)` builds an equality row. `==` is **not** overloaded on `LinExpr`
  — overloading it would break hashing.
- The locking prefix is always extended through the dropoff of anything aboard
  (`_extend_to_whole_pairs`), or a loadable is stranded.
- Empty routes stay in `Solution.routes`; `used_vehicles()` filters them.
- `Capacity` arity must match `CapacitySpace`; mixing arities raises.

## Adding things

**A constraint:** frozen `slots=True` dataclass with `name` property,
`check_route` (or `check_solution`) yielding `Violation`, and `to_model` if
linearisable. Register via
`standard_constraints().with_rules(route_rules=[...])`; do not edit
`standard_constraints`. Worked example: `docs/TUTORIAL.md` §10.

**An objective term:** `name`, `route_cost`, `fleet_cost`, optionally `to_expr`.
Add via `Objective.with_terms` or construct an `Objective` directly.

**An algorithm:** `name` property and `solve(study, initial)`. Use `RunRecorder`
for timing/counting/checkpoints so reported numbers are comparable. Count
evaluations, not just wall-clock. Fix any seed.

**An OD matrix:** implement `duration`, `distance`, `time_dependence`. Implement
`BucketedODMatrix` too if it is piecewise and should be exactly linearisable.

## Conventions

- Google-style docstrings; `ruff` enforces shape (`D` rules on).
- Full annotations; `mypy --strict` must pass. The protocols are the design.
- Frozen `slots=True` dataclasses for value types.
- Comments explain **why**, not what.
- British spelling in prose (`minimise`, `linearisable`, `behaviour`).
- Line length 100.
- Commit messages: imperative present (`Add minimum headway constraint`).

## Verification

Every change must pass all four:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest            # 113 tests
```

Tests needing a solver use `pytest.importorskip("pulp")`; slow ones are marked
`@pytest.mark.slow`.

Required test patterns:

- **Anything that prunes** needs an equivalence test against the unpruned
  enumeration. Pattern: `tests/test_slack.py`.
- **Anything touching a constraint or objective** must keep
  `tests/test_mathprog.py::test_solver_optimum_agrees_with_the_evaluator`
  passing — solver objective and evaluator cost for the same solution must match
  exactly.
- **Randomised algorithms** need a same-seed determinism test.
- Evaluator fixtures use five stops on a line, unit speed, so expected times are
  integers checkable by hand (`tests/conftest.py`).

## Not built — seams left open

| seam | where it goes |
|---|---|
| Loading layout / LIFO for goods | a `RouteConstraint` beside `CapacityLimit` |
| Exact time-dependent MIP | `PiecewiseODMatrix` + `BucketedODMatrix` exist; `PickupDeliveryVars` designed for bucket binaries; `formulation.py` currently requires a frozen profile |
| Skills / compatibility matching | `Vehicle.skills` and `Request.attributes` exist, uninterpreted |
| Parallel candidate evaluation | schedules picklable, evaluation pure; needs a pool wrapper only |
| Subtour elimination beyond big-M | `assignment_to_solution` detects and stops on a disconnected subtour; cut generation would sit beside `build_model` |
| Matheuristics | both round-trip directions already exist |
| Flexible driver breaks | `DriverBreak.flexible_by` stored, treated as pinned |
| Richer intra-route local search | `RouteBuffer.reverse_segment` is the 2-opt primitive, unused |

## References

- Savelsbergh (1992) — forward time slack.
- Cordeau & Laporte (2003) — tabu search for static multi-vehicle dial-a-ride.
- Cordeau (2006) — the three-index formulation `mathprog/formulation.py` follows.
- Ropke & Pisinger (2006) — ruin operators and adaptive weighting in `search/lns.py`.
