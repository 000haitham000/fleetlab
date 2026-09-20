# Architecture

This document does two things: it reviews the design of the original
`Vehicle.java`, and it explains what `fleetlab` does differently and why. Read
it before adding anything to the framework — most of what follows is the
justification for constraints the code deliberately places on you.

---

## Part 1 — Review of `Vehicle.java`

### What the original got right

Four decisions in the Java design were correct and are kept here.

**Timing is derived, not stored.** There was no `computeTimings()` stamping
values onto actions. `getExpectedArrivalTime` walked the action list forward on
every call, so the schedule was a pure function of the route order and the OD
matrix. This eliminates an entire class of bug — the schedule and the route can
never disagree, because there is only one of them.

**The actual/expected duality.** Four action statuses dispatching between
recorded and predicted times is what makes a rolling-horizon study possible at
all. The same code plans a fresh day and re-plans a half-executed one.

**Locking as a first-class concept.** `getLockedActionList` recognised that a
dynamic optimiser free to rewrite everything will thrash a fleet that is already
driving. The rule — freeze the next action if its service falls within
`max(fixed window, fraction × leg)` — is a good rule, and it survives here.

**Injected strategies at the right seams.** `ODMatrix` and
`CapacityLayoutManager` were interfaces, so travel data and capacity semantics
could vary per study without touching the vehicle.

### What blocked the stated goals

The brief was a framework for applying and comparing different algorithms,
extended to cover mathematical programming. Four properties of the design stood
in the way.

#### 1. One class held three responsibilities

`Vehicle` was simultaneously the **route** (it owned `actionList`), the
**schedule evaluator** (`getExpectedArrivalTime` and friends) and the
**constraint checker** (`checkArrivalTimeDiscrepancy`,
`checkAvailabilityEndDiscrepancy`, …). These change for different reasons and,
more importantly, they want opposite mutability — which is the root of most of
what follows.

#### 2. Feasibility required mutating the thing being tested

`addAction` inserted the action, ran the checks against the now-modified list,
and on failure removed it and threw. Three consequences:

- **You cannot evaluate a candidate without risking the state.** Testing 200
  insertion positions meant 200 `SerializationUtils.clone` calls — a full Java
  serialisation round trip each — or accepting that a failed check leaves the
  route in a state you have to trust the unwind to have restored.
- **Rollback was trust-based.** `actionList.remove(action)` removes the *first
  equal* element, and anything else the insertion changed was not undone.
- **No parallel evaluation.** Two threads cannot test two candidate positions on
  one vehicle, because both must install their candidate in the same list.

#### 3. Infeasibility was a boolean

`throw new InconsistentTimingException(...)` says the schedule is infeasible. It
never says **by how much**. Penalty-based metaheuristics — guided local search,
adaptive large neighbourhood search, annealing that deliberately crosses into
infeasible territory — all steer on the margin, and most competitive
pickup-and-delivery heuristics traverse infeasible space on purpose. In a
framework built to compare algorithms, foreclosing that family is the most
expensive item on this list.

Exception construction also ran `String.format` over `toString()` of the vehicle
and the action on every rejected candidate, in the hottest loop in the program.

#### 4. The mathematical-programming lane was not reachable

This is worth stating precisely, because the obvious diagnosis is wrong.

It is **not** that the methods were procedural rather than declarative. It is
that `Vehicle` only ever answered *"given this order, when do we arrive?"* — and
a mathematical program never asks that question. The order is its **output**, not
its input. There is no `actionList` to hand it. The class had nothing to offer
because it was answering a question the solver does not have.

What a solver needs is the **constraint semantics stated independently of any
sequence**, and those semantics existed only as imperative checks buried inside
`addAction`. `throw new CapacityLayoutException` cannot become a row in a
constraint matrix.

The specific obstacle underneath is the time-dependent OD matrix.
`odMatrix.getDuration(departureTime, a, b)` makes travel time a function of
`B_i`, which in a model is a decision variable — so the model is not linear, and
since traffic profiles are not convex it is not rescuable by convexity either.

#### 5. Missing abstractions

- **No objective.** Constraints but no cost. A framework for comparing
  algorithms must state explicitly what "better" means, or the comparison is
  between different problems.
- **No fleet-level solution.** Nowhere for the unassigned pool to live, which
  makes ruin-and-recreate inexpressible and leaves no home for requests a study
  deliberately declines.
- **No onboard time.** Not represented anywhere, so it could be neither
  constrained (the defining dial-a-ride rule) nor costed (the usual
  quality-of-service term).

#### 6. Complexity

`getExpectedArrivalTime` is O(n) from the head of the list. Asking a route for
all its arrival times is O(n²). `checkArrivalTimeDiscrepancyForSucceedingActions`
walks the tail calling `getArrivalTime` per successor, making each insertion
attempt O(n²). Combined with a serialisation-based deep copy per candidate, the
constant factors dominate everything else.

### Concrete defects found

These are carried as regression tests here, named in the test files.

| | defect | consequence |
|---|---|---|
| 1 | `removeRequests` uses `Stream.anyMatch`, which **short-circuits** | removal stops at the first success; the rest silently stay on the route |
| 2 | `getFutureRequests` contradicts its own Javadoc | a request whose pickup is `LOADING` — the vehicle is at the stop loading it — is reported as reassignable to another vehicle |
| 3 | Three competing definitions of "target time" | the `-1` insertion path uses `Utilities.getTargetedDropoffTime`, `checkArrivalTimeDiscrepancy` uses the raw requested time, `calculateActionArrivalTime` uses promised-or-requested; the bracket-finding loop can therefore fail to find a bracket its own bounds check guaranteed, and fall through to appending at the end |
| 4 | `getExpectedServiceTime` ignores `promisedTime` for pickups | a promised time later than the requested time is honoured during arrival propagation and ignored when service starts |
| 5 | `allActionsAreCompleted` inspects only the last element | and throws on an empty list, though it is public |
| 6 | `finalStop` never enters a timing calculation | the availability-end check stops at the last action's departure, so a route that cannot get the vehicle home counts as feasible |
| 7 | `breakList` and `IdleTimeStrategy` are stored but inert | a driver break never pushes a single time |
| 8 | Early-arrival check commented out in `checkArrivalTimeDiscrepancy` | only lateness is enforced; whether that was intended is not recorded |
| 9 | `getDeepCopy` uses `SerializationUtils.clone` | full serialisation round trip per candidate, in the hot loop |

---

## Part 2 — What `fleetlab` does instead

### 2.1 Three objects where there was one

| responsibility | here | mutability |
|---|---|---|
| the route | `domain.Schedule` | immutable |
| the schedule evaluator | `timing.evaluate_route` | pure function |
| the constraint checker | `feasibility.ConstraintSet` | pure function |
| the vehicle itself | `domain.Vehicle` | immutable specification, owns no route |

`Study` bundles the four things that stay fixed while an algorithm runs —
instance, service policy, rules, objective — and is the only object an algorithm
author holds.

### 2.2 Immutability, and the ergonomics objection

The natural objection is that immutability is less convenient. For the *first*
solution you build, that is true. For every iteration after that it is false,
and the reason is that **every metaheuristic needs undo**.

```python
# mutable: rejecting a move requires restoring the previous state
def iteration(sol, rng):
    snapshot = deepcopy(sol)  # required, and not cheap
    removed = ruin(sol, rng)
    try:
        recreate(sol, removed, rng)
    except InfeasibleException:
        sol = snapshot
        return sol
    if not accept(cost(sol), cost(snapshot), rng):
        sol = snapshot
    return sol


# pure: "go back" is not rebinding a name
def iteration(sol, rng):
    partial, removed = ruin(sol, rng)
    cand = recreate(partial, removed, rng)
    return cand if accept(cost(cand), cost(sol), rng) else sol
```

The mutable version was paying for the copy anyway; it just was not visible at
the call site. And the cost comparison runs the other way from intuition: a
`Schedule` over 200 actions is 200 pointers, so building a new one is a memcpy
on the order of a microsecond, against hundreds of microseconds for a
serialisation clone.

Where mutation genuinely reads better, it is available inside a bounded scope:

- `Schedule.editing()` → `RouteBuffer`, for construction and hot intra-route
  loops (it has `reverse_segment`, the 2-opt primitive);
- `Solution.editing()` → `SolutionEditor`, for sequential assignment.

Both commit once to an immutable value. Nothing outside observes a half-built
state, so none of the guarantees are given up.

### 2.3 Violations, not exceptions

```python
@dataclass(frozen=True, slots=True)
class Violation:
    constraint: str
    magnitude: float  # always positive, in natural units
    unit: str  # "minutes", "seats", "kg", "requests"
    vehicle: VehicleId | None
    request: RequestId | None
    index: int | None
    detail: str
```

`Feasibility.penalty(weights)` gives the weighted total a penalised objective
adds to the true cost. Weights are keyed by constraint name and can be adapted
during a run, which is how adaptive LNS escapes infeasible basins.

**Exceptions are reserved for programmer errors** — an out-of-range index, an
unknown vehicle id, a plan that rewrites history. Never for modelling outcomes.
This rule is in `CONTRIBUTING.md` because it is the one most likely to erode.

### 2.4 Constraints with two faces

The change that makes one framework serve both lanes:

```python
class RouteConstraint(Protocol):
    def check_route(self, schedule, timing, ctx) -> Iterator[Violation]: ...


class Linearisable(Protocol):
    def to_model(self, model, variables, ctx) -> None: ...
```

A constraint that cannot be linearised simply does not implement
`Linearisable`, and `ConstraintSet.non_linearisable()` names it. The exporter
then reports that the model is a **relaxation** rather than silently producing a
bound for a looser problem — the kind of error that survives into a paper
unnoticed.

The same applies to objective terms via `LinearObjectiveTerm`, and
`BuildReport.is_exact` is the single flag to check before quoting an optimum as
anything other than a bound.

### 2.5 Time dependence, stated honestly

`ODMatrix` must declare its own `TimeDependence`:

| | meaning | mathematical programming |
|---|---|---|
| `CONSTANT` | fixed matrix | exact |
| `PIECEWISE` | constant within time buckets | exact for the discretisation |
| `ARBITRARY` | opaque callable | simulation only |

`build_model` **refuses** a non-constant matrix unless given an explicit
`freeze_profile_at`, and the resulting `BuildReport` records the approximation
and states that its optimum is not a valid bound. `PiecewiseODMatrix` also has
`fifo_violations()`, which reports bucket boundaries where leaving later would
let a vehicle arrive earlier — a data problem that otherwise shows up as a
suspiciously good schedule.

### 2.6 Forward time slack — and the bug it caught

An insertion scan asks, for each candidate position, whether the delay it
introduces breaks something later. Answered naively that is O(n) per candidate.
Forward time slack (Savelsbergh 1992; used throughout Cordeau & Laporte's
dial-a-ride work) precomputes in one backward O(n) pass how much delay each
position can absorb, making each candidate a comparison.

**This is the concrete reason the route had to become immutable.** The slack
array is derived from a route's timing; in a design where testing a candidate
means mutating the route, the mutation invalidates the array you were about to
test against.

The screen is a **filter, not a verdict** — exact only for a constant matrix
with no onboard cap, otherwise an optimistic screen whose survivors are
confirmed by full evaluation. Optimistic is the safe direction: it may pass a
candidate that turns out infeasible, never reject one that would have worked.

During development the equivalence test in `tests/test_slack.py` caught exactly
the failure it was written for. `insertion_push` treated a vehicle's depot
departure as fixed, but under the just-in-time rule inserting a new *first*
action lets the vehicle leave **earlier** — so nothing downstream is pushed at
all. Every position-zero candidate was being discarded despite being feasible,
and often the best available. The fix routes both the evaluator and the screen
through one `depot_departure_for`, and the test asserts that screened and
unscreened enumeration produce identical candidate sets.

### 2.7 People and goods, without a type check

Capacity is a **vector** over named dimensions:

```python
space = CapacitySpace(("seats", "wheelchair", "kg", "volume"))
passenger = space.of(seats=1)
wheelchair = space.of(wheelchair=1)
parcel = space.of(kg=12.5, volume=0.08)
```

A `Loadable` carries a demand vector and a free-form `kind` label used for
reporting only. **The core never branches on `kind`.** A goods-only van has
`seats=0`, so the capacity constraint rejects passengers with no type check
anywhere. Violations report the offending dimension in its own unit, so a
penalty function can weight a mass overflow differently from a seat overflow.

`max_onboard_time` is optional per request. Setting it makes a request
dial-a-ride; leaving it `None` makes it goods. That, plus the pluggable
`ConstraintSet`, is the whole mechanism by which one framework covers both.

Vocabulary follows: `Loadable`, `service_duration`, `dwell`, `onboard_time` —
never passenger, boarding, or ride time.

### 2.8 Simulation and search want opposite things

| activity | wants | because |
|---|---|---|
| simulation | mutable | one timeline, monotone progress; the clock advances and a vehicle arrives |
| search | immutable | millions of throwaway candidates, each discardable for free |

The Java `Vehicle` did both, which is why `currentTime`, `currentStop` and the
action statuses sat so uneasily beside `addRequest`.

```python
while sim.advance_to_next_decision_epoch():
    ctx = sim.freeze()  # boundary: immutable snapshot
    plan = optimiser.solve(ctx, sim.plan)  # pure
    sim.commit(plan)  # mutable
```

`ExecutionState.commit` refuses a plan that reorders or drops an action already
performed, so an optimiser bug cannot produce a simulation that could not have
happened. Locking became a `LockingPolicy` returning a committed **prefix
length** (`HorizonLocking` reproduces the Java rule; `NoLocking` and
`LockOnboard` are alternatives), and the prefix is always extended through the
dropoff of anything aboard — a prefix ending between a pickup and its dropoff
would strand the loadable.

### 2.9 Smaller decisions worth recording

**Minutes as floats, not `ZonedDateTime`.** A model needs numbers; datetime
arithmetic cannot appear in a constraint row. Wall-clock conversion happens only
at the IO boundary (`Epoch`). Float arithmetic is also roughly two orders of
magnitude cheaper, and the evaluator is the innermost loop.

**Absolute time windows, not offsets.** The Java stored tolerances around a
requested time and resolved them at each use site against different base times.
Storing the resolved `[earliest, latest]` kills that bug class and is exactly the
`e_i`, `l_i` a model needs. `window_around()` is there for instances authored as
offsets.

**One definition of target time.** `Request.target_time` is promised-if-given,
else requested. Nothing else may form its own answer.

**`committed` as an integer prefix**, replacing the per-action status scan. "May
I insert here?" is an integer comparison.

**The service policy is injected.** "Wait at a pickup, serve immediately at a
dropoff" is a modelling decision that differs between studies and determines the
shape of the corresponding model rows. `EarlyArrivalPolicy` is the default;
`PunctualPolicy` is the alternative.

**The shared-stop dwell discount defaults to off.** The Java halved service time
when an action shared a stop with the previous one. That is a substantive
assumption, so `shared_stop_factor` defaults to `1.0` and a study opts in.

**`finalStop` and driver breaks are now honoured.** The return leg is evaluated
and counted against the shift end; breaks push service.

### 2.10 Performance

- Evaluation: one O(n) pass per route, memoised on the schedule's hash.
- Insertion feasibility: O(1) per position via cached forward slack.
- No deep copies anywhere — immutable structures are shared.
- MIP: per-arc big-M from the two nodes' windows rather than a global constant
  (a loose big-M is the most common reason a correct pickup-and-delivery
  formulation refuses to solve), plus time-window arc pruning.

---

## Part 3 — Seams left open

Deliberately not built, with the hook already in place.

| seam | where it goes |
|---|---|
| Loading layout / LIFO for goods | a `RouteConstraint` beside `CapacityLimit`; the vector capacity already handles the quantity half |
| Exact time-dependent MIP | `PiecewiseODMatrix` and `BucketedODMatrix` exist and the variable bundle is designed for bucket-selection binaries; `formulation.py` currently requires a frozen profile |
| Skills / compatibility matching | `Vehicle.skills` and `Request.attributes` exist and are uninterpreted; add a `RouteConstraint` |
| Parallel candidate evaluation | schedules are picklable and evaluation is pure; needs a worker-pool wrapper, no design change |
| Subtour elimination beyond big-M | `assignment_to_solution` detects a disconnected subtour and stops rather than looping; a cut-generation loop would go beside `build_model` |
| Matheuristics | `solution_to_assignment` and `assignment_to_solution` are both present, which is what a solve-over-a-subset loop needs |
| Flexible driver breaks | `DriverBreak.flexible_by` is stored; the policy currently treats every break as pinned |
| Richer intra-route local search | `RouteBuffer.reverse_segment` is the 2-opt primitive; no operator uses it yet |

## References

- Savelsbergh (1992), *The vehicle routing problem with time windows: minimizing
  route duration* — forward time slack.
- Cordeau & Laporte (2003), *A tabu search heuristic for the static multi-vehicle
  dial-a-ride problem*.
- Cordeau (2006), *A branch-and-cut algorithm for the dial-a-ride problem* — the
  three-index formulation this one follows.
- Ropke & Pisinger (2006), *An adaptive large neighbourhood search heuristic for
  the pickup and delivery problem with time windows* — the ruin operators and
  adaptive weighting in `search/lns.py`.
