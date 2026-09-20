# fleetlab

A framework for **simulation-based optimisation of pickup-and-delivery problems**,
built so that different algorithms can be applied to the same problem and
compared honestly.

What is being moved may be people or goods. The core never asks: a `Loadable` is
whatever occupies capacity, and capacity is a vector over named dimensions —
seats, wheelchair bays, kilograms, cubic metres. A dial-a-ride study is the
configuration where per-request onboard-time caps are active; a parcel study is
the one where they are not. There is no `if moving_people` anywhere in the
framework.

## The one idea

The framework has two lanes over **one** problem statement:

| lane | the sequence is | the question |
|---|---|---|
| **search** | known | given this order, is it feasible and what does it cost? |
| **mathematical programming** | unknown | which order? |

Every constraint and every objective term states itself in **both**:

```python
class TimeWindows:
    def check_route(self, schedule, timing, ctx):  # search lane
        for stop in timing.stops:
            lateness = window_of(stop).lateness(stop.service_start)
            if lateness > 0:
                yield Violation("time_window", magnitude=lateness, unit="minutes")

    def to_model(self, model, variables, ctx):  # MIP lane
        for node in ctx.nodes:
            model.add(variables.service_start[node] >= earliest[node])
            model.add(variables.service_start[node] <= latest[node])
```

So a bound and a heuristic are provably answering the same question. The test
suite asserts it directly: solve the model, read the answer back as an ordinary
`Solution`, and re-evaluate it with the same evaluator every heuristic uses. The
two numbers must agree to the last digit.

## Install

```bash
uv sync --all-extras --dev      # or: pip install -e ".[dev]"
```

Python 3.11+. The mathematical-programming lane has **no solver dependency** —
it writes standard LP files. Install the `solvers` extra only if you want to
solve in process.

## Five minutes

```python
from fleetlab.study import Study
from fleetlab.io.generate import mixed_instance
from fleetlab.search import RegretInsertion, AdaptiveLNS

problem = mixed_instance(passengers=8, wheelchair_users=2, parcels=6, vehicles=3)
study = Study(problem)  # instance + rules + objective

built = RegretInsertion().solve(study)
print(built.describe())

improved = AdaptiveLNS(iterations=600).solve(study, built.solution)
print(improved.breakdown.describe())  # per-term cost, not one number
```

An exact bound for the same study:

```python
from fleetlab.mathprog import build_model, solution_to_assignment
from fleetlab.mathprog.adapters import solve_with_pulp, write_lp

report = build_model(study)
print(report.describe())  # size, pruning, and whether it is exact

write_lp(report.model, "instance.lp")  # hand to any solver
# ...or solve in process, warm-started from the heuristic:
warm = solution_to_assignment(study, report.variables, improved.solution)
solved = solve_with_pulp(report.model, time_limit=300, warm_start=warm)
```

A simulated day, where requests arrive over time:

```python
from fleetlab.simulation import Simulator, HorizonLocking

result = Simulator(cadence=30.0, locking=HorizonLocking()).run(study, RegretInsertion())
print(result.describe())
```

Note that `RegretInsertion` is unchanged between the static and dynamic cases.
It never learns that a clock exists — see *Simulation* below.

Run `python examples/run_study.py` for all three together.

## Layout

```
fleetlab/
├── domain/        entities: Stop, Loadable, Request, Vehicle, Schedule, Solution, Problem
├── od/            travel matrices, each declaring its time dependence
├── timing/        the forward pass: evaluator, service policy, forward time slack
├── feasibility/   constraints — each with a check face and a to_model face
├── objective/     decomposable, weighted, reportable cost terms
├── moves/         neighbourhood steps as data; insertion and ruin operators
├── search/        the heuristic lane: Algorithm protocol, construction, adaptive LNS
├── mathprog/      variables, structural rows, round trip, optional solver adapters
├── simulation/    the mutable half: clock, execution state, locking, decision epochs
├── io/            JSON instances and synthetic generation
└── study.py       Study — the one object an algorithm author holds
```

## Four design decisions worth knowing before you read the code

**Timing is derived, never stored.** One forward pass over a route produces
arrivals, service starts, departures, waits, onboard times, the load profile and
the return leg. There is no cached timing to invalidate, so a move is just a
list edit.

**A schedule is immutable.** Rejecting a candidate costs nothing — it is simply
not rebinding a name. That is what makes evaluation cacheable, parallelisable,
and, most importantly, what keeps a route's forward time slack valid so an
insertion feasibility test is O(1) instead of O(n). Convenience layers
(`Schedule.editing()`, `Solution.editing()`, `RouteBuffer`) give back the
ergonomics of mutation where sequential construction wants it.

**Infeasibility is data, not an exception.** A check returns `Violation` objects
carrying a magnitude in natural units — minutes late, seats over, kilograms
over. Exceptions are reserved strictly for programmer errors. A boolean verdict
would foreclose every penalty-based metaheuristic, which is most of the good
ones.

**Simulation is mutable; search is not.** `ExecutionState` owns the clock and
the fleet's real positions and mutates freely. The optimiser only ever sees a
frozen snapshot. Splitting these was the single most valuable change from the
earlier Java design, where one `Vehicle` object did both jobs.

`ARCHITECTURE.md` explains each of these against the Java original it replaces,
including the bugs that motivated them. `CONTRIBUTING.md` has the rules a new
constraint, objective term or algorithm has to follow.

## Status

Alpha. The spine runs end to end and is tested, but the algorithms that ship are
a baseline and a worked reference, not competitors — a study's own algorithms are
the point. Known seams are listed at the end of `ARCHITECTURE.md`.
