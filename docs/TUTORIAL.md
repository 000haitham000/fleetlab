# A tour of fleetlab

This guide is for someone who has never seen this code before. It takes about
twenty minutes. By the end you will understand what the main pieces are, how
they fit together, and you will have added a rule of your own.

You need to know Python. You do not need to know anything about vehicle routing
— everything is explained as it comes up.

Every code block here can be pasted into a Python file and run. They build on
each other, so read them in order.

---

## 1. The problem this solves

Imagine you run a small fleet of minibuses in a town.

People phone up and ask to be taken from one place to another. Parcels need
collecting from a warehouse and dropping at shops. Each vehicle can carry
several of these at once, so it makes sense to combine them into one trip.

Your job is to decide **which vehicle does which job, and in what order**.

That is harder than it sounds. A passenger who is picked up first might sit in
the bus for forty minutes while the driver collects three other people. A parcel
might be delivered to a shop that has already closed. A route that looks
efficient might not get the driver back to the depot before their shift ends.

This kind of problem has a name: a **pickup-and-delivery problem**. Every job
has two parts — a place to collect from and a place to deliver to — and they must
be done by the same vehicle, in that order.

`fleetlab` gives you the pieces to describe such a problem, to check whether a
proposed plan is any good, and to search for better plans.

### One vocabulary for people and for parcels

A passenger and a parcel are different things in real life. In this code they
are the same thing: something that takes up room in a vehicle between a pickup
and a dropoff. The word used for it is **loadable**.

This is not laziness. It means there is no code anywhere that asks "is this a
person or a box?" — which in turn means you can mix people and goods on the same
vehicle without the framework needing a special case for it.

The difference between them shows up in the *data*, not in the code:

- a passenger takes up one **seat**
- a parcel takes up **kilograms** and **cubic metres**
- a van with zero seats simply cannot carry a passenger

More on that in a moment.

---

## 2. Describing a problem

Six things are needed. We will build each one.

### Stops — places

```python
from fleetlab.domain import Stop, StopId

depot = Stop(StopId("depot"), 0.0, 0.0, "depot")
library = Stop(StopId("library"), 3.0, 0.0, "library")
clinic = Stop(StopId("clinic"), 9.0, 0.0, "clinic")
warehouse = Stop(StopId("warehouse"), 6.0, 0.0, "warehouse")
shop = Stop(StopId("shop"), 12.0, 0.0, "shop")
```

Those numbers are x and y coordinates. To keep the arithmetic easy, all five
stops sit on a straight line, three units apart. Later we will set the vehicle's
speed to one unit per minute — so a gap of 3 units takes exactly 3 minutes. That
way you can check every number in this guide in your head.

### Capacity — how room is measured

Room in a vehicle is not one number. A minibus might have four seats and a
luggage limit. A van might have no seats at all but plenty of weight allowance.

So capacity is a **list of named amounts**. You choose the names once, at the
start:

```python
from fleetlab.domain import CapacitySpace

space = CapacitySpace(("seats", "kg"))
```

Now `space.of(seats=1)` means one seat and no weight. `space.of(kg=12)` means
twelve kilograms and no seat.

### Loadables — the things being moved

```python
from fleetlab.domain import Loadable, LoadableId

amina = Loadable(LoadableId("amina"), space.of(seats=1), kind="passenger")
package = Loadable(LoadableId("pkg-1"), space.of(kg=12), kind="parcel")
```

`kind` is only a label for reports and charts. Nothing in the framework reads
it. What actually distinguishes a passenger from a parcel is that one takes a
seat and the other takes kilograms.

### Requests — the jobs

A **request** is one job: move these loadables from here to there.

```python
from fleetlab.domain import Request, RequestId, TimeWindow

ride = Request(
    id=RequestId("ride"),
    origin=StopId("library"),
    destination=StopId("clinic"),
    loadables=(amina,),
    pickup_window=TimeWindow(10.0, 40.0),  # collect between 00:10 and 00:40
    dropoff_window=TimeWindow(20.0, 90.0),  # deliver between 00:20 and 01:30
    requested_pickup=20.0,  # she asked for 00:20
    pickup_service_duration=2.0,  # 2 minutes to board
    dropoff_service_duration=2.0,  # 2 minutes to get off
    max_onboard_time=60.0,  # no more than an hour in the bus
)

delivery = Request(
    id=RequestId("delivery"),
    origin=StopId("warehouse"),
    destination=StopId("shop"),
    loadables=(package,),
    pickup_window=TimeWindow(15.0, 60.0),
    dropoff_window=TimeWindow(25.0, 120.0),
    requested_pickup=30.0,
    pickup_service_duration=3.0,
    dropoff_service_duration=3.0,
    # no max_onboard_time: a parcel does not mind a long trip
)
```

Times are **minutes from the start of the day**, not clock times. So `20.0`
means twenty minutes in. This keeps the arithmetic simple and, as you will see
later, it is also what a mathematical solver needs.

A **time window** is the period during which service may begin. Two pieces of
vocabulary that often get confused:

- **arrival** is when the vehicle reaches the stop
- **service start** is when the work actually begins

They are different whenever the vehicle turns up early and has to wait.

Notice that `ride` has a `max_onboard_time` and `delivery` does not. That single
field is the whole difference between a dial-a-ride study and a parcel study.
Passengers mind how long they are in the vehicle. Parcels usually do not.

### Vehicles

```python
from fleetlab.domain import Vehicle, VehicleId

minibus = Vehicle(
    id=VehicleId("bus-1"),
    start_stop=StopId("depot"),
    end_stop=StopId("depot"),  # it must come back
    capacity=space.of(seats=4, kg=500),
    available_from=0.0,
    available_until=300.0,  # a five-hour shift
)
```

A `Vehicle` here describes what the vehicle *is*. It does not hold a route. The
route is a separate thing, and section 8 explains why that separation matters so
much.

### The problem — everything together

```python
from fleetlab.domain import Problem
from fleetlab.od import EuclideanODMatrix

stops = [depot, library, clinic, warehouse, shop]

problem = Problem.build(
    name="tutorial",
    capacity_space=space,
    stops=stops,
    requests=[ride, delivery],
    vehicles=[minibus],
    od=EuclideanODMatrix(stops, speed=1.0),
    horizon_end=300.0,
)
```

The **OD matrix** ("origin–destination matrix") says how long it takes to get
from any stop to any other. `EuclideanODMatrix` just measures straight-line
distance and divides by the speed — fine for a tutorial. A real study would load
measured travel times instead.

---

## 3. A route, and what the framework works out about it

A route is called a **`Schedule`**. It is a list of **actions**, in the order
they happen. An action is one half of one request: either the pickup or the
dropoff.

Let us write one out by hand:

```python
from fleetlab.domain import Schedule

pickup_ride, dropoff_ride = problem.actions_for(RequestId("ride"))
pickup_delivery, dropoff_delivery = problem.actions_for(RequestId("delivery"))

route = Schedule(
    VehicleId("bus-1"),
    (pickup_ride, pickup_delivery, dropoff_ride, dropoff_delivery),
)

print(route.describe())
# bus-1: P(ride) P(delivery) D(ride) D(delivery)
```

The bus collects Amina, then the package, then drops Amina off, then the
package. Both are in the vehicle at the same time in the middle. That is the
whole point of ride-sharing.

Now the interesting part. Ask the framework what this route actually looks like
in time:

```python
from fleetlab.timing import EvalContext, evaluate_route

ctx = EvalContext(problem)
timing = evaluate_route(route, ctx)

print(timing.describe())
```

You get:

```
bus-1: depart 00:17, return 00:56, drive 24.0m, wait 5.0m
  #0 arr 00:20 svc 00:20 dep 00:22
  #1 arr 00:25 svc 00:30 dep 00:33
  #2 arr 00:36 svc 00:36 dep 00:38
  #3 arr 00:41 svc 00:41 dep 00:44
```

Read that line by line. Every number can be checked by hand.

**The bus leaves the depot at 00:17.** Not at 00:00. Amina asked to be collected
at 00:20, and the library is 3 minutes away, so the bus leaves at exactly the
right moment. Leaving earlier would only mean sitting at the library doing
nothing.

**Action 0, collecting Amina.** Arrives 00:20, starts straight away, takes 2
minutes to board, leaves at 00:22.

**Action 1, collecting the package.** The warehouse is 3 minutes from the
library, so the bus arrives at 00:25. But the package was requested for 00:30.
So the bus **waits 5 minutes**, then spends 3 minutes loading, and leaves at
00:33.

**Action 2, dropping Amina at the clinic.** Three minutes away, so it arrives at
00:36. Here is a rule worth remembering: **a vehicle that arrives early at a
pickup waits, but a vehicle that arrives early at a dropoff gets on with it.**
The reasoning is that a passenger might not be ready before the agreed time, but
once you have arrived somewhere there is no reason not to let them out. So
service starts at 00:36 immediately.

**Action 3, the package at the shop.** Arrives 00:41, done by 00:44.

**Back at the depot by 00:56.** The shop is 12 minutes from the depot. That
return trip is part of the route — a plan that gets all the work done but strands
the driver twenty minutes from home at the end of their shift is not a good plan,
and the framework will tell you so.

### Things the timing knows

```python
print(timing.onboard_time(RequestId("ride")))  # 14.0
print(timing.excess_onboard_time(RequestId("ride")))  # 8.0
print(timing.total_wait)  # 5.0
```

**Onboard time** is how long Amina was in the bus: she left the library at 00:22
and was served at the clinic at 00:36, so 14 minutes.

**Excess onboard time** is how much of that was detour. Going straight from
library to clinic takes 6 minutes. She took 14. So collecting the package cost
her 8 extra minutes.

That number matters. It is usually the thing you are trading off against fuel
and driver time, and it is the reason someone complains about your service.

You can also see what the vehicle was carrying at each moment:

```python
for i, load in enumerate(timing.loads):
    print(i, space.describe(load))
# 0 seats=1
# 1 seats=1 kg=12
# 2 kg=12
# 3 empty
```

---

## 4. Rules

A plan can be worked out in time and still be no good. Maybe it delivers after
the shop closes. Maybe it puts five people in a four-seat bus.

Those are **constraints**. Checking them looks like this:

```python
from fleetlab.feasibility import standard_constraints

rules = standard_constraints()
report = rules.check_route(route, timing, ctx)

print(report.describe())
# feasible
```

Now break something on purpose. Swap the order so the dropoff comes before the
pickup:

```python
broken = Schedule(VehicleId("bus-1"), (dropoff_ride, pickup_ride))
broken_timing = evaluate_route(broken, ctx)
print(rules.check_route(broken, broken_timing, ctx).describe())
```

```
1 violation(s):
  precedence: 2 positions over [bus-1 ride #0] -- dropoff at #0 precedes pickup at #1
```

### Two things to notice

**Nothing raised an exception.** Checking a bad plan is a completely normal
thing to do, not an error. The function returns a list of what is wrong.

**Each problem comes with a size.** Not "this plan is invalid" but "this plan is
7 minutes late" or "this plan is 2 seats over". That number is called the
**magnitude**, and it is reported in its own natural unit.

The size matters more than you might expect. Many good search algorithms work by
deliberately allowing slightly-broken plans, then steering back towards a valid
one. To steer, they need to know whether they are getting closer or further
away. A plain yes/no answer gives them nothing to steer by.

```python
print(report.penalty())  # 0.0 — nothing wrong
print(rules.check_route(broken, broken_timing, ctx).penalty())
```

---

## 5. Cost

Rules say what is *allowed*. An **objective** says what is *better*.

```python
from fleetlab.objective import standard_objective

objective = standard_objective()
print(objective)
# Objective(1.0*travel_distance + 0.5*excess_onboard + 100.0*fleet_size + 10000.0*unserved)
```

An objective is a weighted sum of separate terms. This one says: distance costs
something, making passengers detour costs half as much per minute, using another
vehicle costs a lot, and failing to serve a request costs a great deal more.

Those weights are a statement about your priorities. Change them and you get
genuinely different answers — that is the point, not a flaw.

Importantly, you can always see the breakdown:

```
total 63518.6180
  excess_onboard              38.5784  (raw 77.1568)
  fleet_size               33000.0000  (raw 330.0000)
  travel_distance            480.0396  (raw 480.0396)
  unserved                 30000.0000  (raw 3.0000)
```

`raw` is the real quantity — 77 minutes of detour, 3 unserved requests. The other
column is that quantity after weighting. When you compare two algorithms, this
tells you *why* one won, not just that it did.

---

## 6. Study — the object you will actually use

Carrying `problem`, `ctx`, `rules` and `objective` around separately gets old
fast. A **`Study`** holds all four:

```python
from fleetlab.study import Study

study = Study(problem)  # uses the standard rules and objective

timing = study.timing(route)  # when things happen
report = study.check_route(route)  # what is broken
cost = study.route_cost(route)  # what it costs
```

A `Study` is also what makes a fair comparison possible. Two algorithms handed
the same `Study` are definitely solving the same problem, with the same rules,
judged by the same yardstick. If each algorithm built its own setup, any
difference in the results might just be a difference in the setup.

---

## 7. Letting an algorithm do the work

So far we wrote the route by hand. Now let something find one.

```python
from fleetlab.io.generate import mixed_instance
from fleetlab.search import RegretInsertion, AdaptiveLNS

problem = mixed_instance(passengers=8, wheelchair_users=2, parcels=6, vehicles=3)
study = Study(problem)

first = RegretInsertion().solve(study)
print(first.describe())

better = AdaptiveLNS(iterations=600).solve(study, first.solution)
print(better.describe())
```

`RegretInsertion` builds a solution from nothing by placing one request at a
time. `AdaptiveLNS` then improves it by repeatedly tearing out part of the
solution and rebuilding that part differently, keeping the result when it is
better.

Every algorithm has the same shape:

```python
result = algorithm.solve(study, starting_solution)
```

and every one returns the same kind of result — the solution, the cost, the
breakdown, how many iterations it ran, how long it took. That uniformity is
deliberate: it is what lets you put them in a table next to each other.

---

## 8. Two ideas that explain the shape of the code

If two things about this codebase seem odd at first, it is probably these. Both
are deliberate and both buy something specific.

### Idea one: nothing is ever modified

When you change a route, you do not edit it. You get a new one back:

```python
longer = route.with_inserted(some_action, 2)  # route itself is unchanged
```

This seems wasteful. It is not, and here is why.

A search algorithm spends its life asking "what if I did this?" — thousands of
times a second. Most of the answers are "that would be worse", and it throws
them away.

If routes could be edited, then trying something out means changing the real
route, looking at the result, and then putting it back exactly as it was if you
do not like it. Putting it back correctly, every time, for every kind of change,
is fiddly and is where bugs live. The usual fix is to copy the route before every
attempt — which costs far more than just building a new one.

When nothing is modified, "putting it back" is not an operation at all. You just
use the old route, which was never touched. Rejecting an idea costs literally
nothing.

Two more things follow from it. Routes can be safely shared between several
processors working at once, because none of them can interfere with the others.
And results worked out for a route stay correct, because the route cannot change
underneath them.

When you genuinely do want to build something up step by step, there is a
shortcut that does not give any of this away:

```python
with route.editing() as buffer:  # a private scratch copy
    buffer.insert_pair(pickup, 0, dropoff, 1)
    buffer.insert_pair(p2, 1, d2, 3)
new_route = buffer.commit()  # one finished route at the end
```

### Idea two: every rule speaks two languages

There are two ways to attack a routing problem.

**Search.** Guess a route, measure it, try to improve it. Fast, works on big
problems, but you never know if a better answer exists.

**Mathematical programming.** Hand the whole problem to a solver as a set of
equations and let it find the provably best answer. Slow, only works on small
problems, but the answer comes with a guarantee.

Both are useful, and a study usually wants both — the exact answer on a small
version to find out how good the fast method really is.

The trouble is that the two need the rules expressed completely differently.
Search needs "here is a route, tell me what is wrong with it". A solver needs
"here are equations the answer must satisfy" — because the solver is *choosing*
the route, so there is no route to inspect yet.

Write those separately and you have written the rules twice. The two copies will
drift apart, and then you are comparing two different problems without knowing
it.

So here, every rule says the same thing in both languages, in one place:

```python
class TimeWindows:
    def check_route(self, schedule, timing, ctx):
        """For search: the route is known, so measure it."""
        ...

    def to_model(self, model, variables, ctx):
        """For the solver: the route is unknown, so constrain it."""
        ...
```

The test suite checks this really holds. It solves the equations, converts the
solver's answer back into an ordinary route, and re-measures it with the same
code every search algorithm uses. The two numbers have to match exactly.

---

## 9. A day that unfolds

Everything so far assumed you know all the jobs in advance. Often you do not —
people phone up during the day.

```python
from fleetlab.simulation import Simulator

result = Simulator(cadence=30.0).run(study, RegretInsertion())
print(result.describe())
```

This runs a simulated day. Every so often it stops, looks at what has actually
happened, lets the algorithm re-plan, and carries on.

The thing worth noticing: `RegretInsertion` is exactly the same object as in
section 7. It was not modified to work here. It has no idea a clock exists.

That works because of a clean split. The simulator holds the clock and the real
positions of the vehicles, and it changes as time passes. The algorithm only
ever receives a frozen snapshot — a description of where everything is right now
— and hands back a plan. It cannot reach into the simulation, and the simulation
does not care how it came up with the plan.

One more piece: **locking**. If the algorithm were free to change everything
every time, it would send a bus towards one stop, change its mind, and redirect
it — over and over, while the bus is already driving. So whatever is about to
happen is frozen and not offered up for re-planning.

---

## 10. Now build something: your own rule

Suppose drivers complain about having too many jobs crammed into one shift. You
want a rule: **no vehicle may serve more than N requests**.

Every rule is a small class with two methods. Create `max_requests.py`:

```python
"""A rule capping how many requests one vehicle may serve."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from fleetlab.feasibility import Violation
from fleetlab.linear import LinExpr


@dataclass(frozen=True, slots=True)
class MaxRequestsPerRoute:
    """No vehicle may serve more than `limit` requests.

    Attributes:
        limit: The largest number of requests allowed on one route.
    """

    limit: int = 5

    @property
    def name(self) -> str:
        """The label used in reports and penalty weights."""
        return "max_requests"

    def check_route(self, schedule, timing, ctx) -> Iterator[Violation]:
        """For search: count what is on the route and report the excess."""
        count = len(schedule.requests())
        if count <= self.limit:
            return
        yield Violation(
            constraint=self.name,
            magnitude=float(count - self.limit),
            unit="requests",
            vehicle=schedule.vehicle,
            detail=f"{count} requests against a limit of {self.limit}",
        )

    def to_model(self, model, variables, ctx) -> None:
        """For the solver: cap the pickups each vehicle may leave."""
        pickups = [variables.nodes.pickup[r] for r in ctx.problem.requests]
        for vehicle_id in ctx.problem.vehicles:
            served = LinExpr.sum(
                arc for node in pickups for arc in variables.out_of(node, vehicle_id)
            )
            model.add(served <= float(self.limit), f"max_requests_{vehicle_id}")
```

Walk through what each method does.

`check_route` is the search side. The route is sitting right there, so just
count the requests on it. If it is over the limit, describe the problem: which
rule, by how much, in what unit, on which vehicle. If it is not over, say
nothing — no news is good news.

`to_model` is the solver side. There is no route to count, so instead you write
down a restriction the answer must obey. `variables.out_of(node, vehicle_id)`
gives the yes/no decisions for "does this vehicle leave that pickup point?".
Adding them up gives the number of requests the vehicle serves. That total must
not exceed the limit. One line of maths per vehicle.

Now use it:

```python
from fleetlab.feasibility import standard_constraints
from fleetlab.study import Study
from fleetlab.io.generate import mixed_instance
from fleetlab.search import RegretInsertion

from max_requests import MaxRequestsPerRoute

problem = mixed_instance(passengers=8, parcels=6, vehicles=3)

rules = standard_constraints().with_rules(route_rules=[MaxRequestsPerRoute(limit=4)])
study = Study(problem, constraints=rules)

result = RegretInsertion().solve(study)
for schedule in result.solution.routes:
    print(schedule.vehicle, len(schedule.requests()), "requests")
```

No route will have more than four. Change the limit to 3 and run again — the
routes get shorter and more requests go unserved, because there is less room.

Two things to note about how you did that.

You did not edit any existing file. Rules are registered, not hard-coded, so
your study can enforce rules that nobody else's study has.

And you wrote the rule once. Both the search side and the solver side now know
about it, so an exact bound computed for this study will respect your limit too.

---

## 11. Where to look next

| to learn about | look at |
|---|---|
| how a route turns into times | `src/fleetlab/timing/evaluator.py` |
| the rules that ship | `src/fleetlab/feasibility/` |
| how an algorithm is structured | `src/fleetlab/search/greedy.py` |
| a fuller algorithm | `src/fleetlab/search/lns.py` |
| building the equations for a solver | `src/fleetlab/mathprog/formulation.py` |
| how a simulated day runs | `src/fleetlab/simulation/simulator.py` |

Most files open with a comment explaining why the code is shaped the way it is,
not just what it does. Those are worth reading.

`AGENTS.md` in the project root is a condensed reference. It is written for AI
assistants, but it also works as a quick lookup once you already know your way
around.

### A few conventions that will save you confusion

**Times are minutes, stored as ordinary numbers.** `08:30` in the morning might
be `510.0`. There are no date objects anywhere in the core.

**Capacity is always a list of amounts**, never a single number, even when your
problem only has one dimension.

**An exception means someone made a programming mistake** — a bad index, an
unknown id. It never means "this plan is no good". That is always returned as
data.

**Both halves of a request always travel together.** There is no way to move a
pickup to another vehicle and leave its dropoff behind. The design does not
allow it, so you cannot do it by accident.
