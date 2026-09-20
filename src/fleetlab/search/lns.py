"""Adaptive large neighbourhood search.

Ruin part of the solution, rebuild it, keep the result if an acceptance rule
says so. Repeat. The adaptive part is that operators which have been producing
improvements get chosen more often.

This is included as a worked reference rather than as a finished competitor: it
is the shape a study's own metaheuristic should take, and it exercises every
seam in the framework -- ruin operators, insertion, delta evaluation, penalised
cost, the run recorder. Read it as the worked example in
:doc:`CONTRIBUTING <contributing>` intends.

Note how short the iteration is. There is no snapshot and no undo, because
rejecting a candidate is simply not rebinding a name -- the incumbent was never
touched. That is the ergonomic payoff of the immutable core, and it is why the
"mutation is more convenient" intuition does not survive contact with an
acceptance criterion.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fleetlab.moves.ruin import (
    random_removal,
    related_removal,
    route_removal,
    worst_removal,
)
from fleetlab.search.base import RunRecorder
from fleetlab.search.greedy import RegretInsertion

if TYPE_CHECKING:
    from fleetlab.domain.solution import Solution
    from fleetlab.moves.base import ReleaseRequests
    from fleetlab.search.base import RunResult
    from fleetlab.study import Study

RuinOperator = Callable[["Study", "Solution", int, random.Random], "ReleaseRequests"]


def _random(study: Study, solution: Solution, count: int, rng: random.Random) -> ReleaseRequests:
    del study
    return random_removal(solution, count, rng)


def _related(study: Study, solution: Solution, count: int, rng: random.Random) -> ReleaseRequests:
    return related_removal(study, solution, count, rng)


def _worst(study: Study, solution: Solution, count: int, rng: random.Random) -> ReleaseRequests:
    return worst_removal(study, solution, count, rng)


def _whole_route(
    study: Study, solution: Solution, count: int, rng: random.Random
) -> ReleaseRequests:
    del study, count
    return route_removal(solution, rng)


DEFAULT_OPERATORS: tuple[tuple[str, RuinOperator], ...] = (
    ("random", _random),
    ("related", _related),
    ("worst", _worst),
    ("route", _whole_route),
)


@dataclass
class OperatorScores:
    """Adaptive weights over ruin operators.

    Operators are scored by what they achieve -- a new global best is worth more
    than a merely accepted move -- and weights decay toward recent performance
    rather than averaging over the whole run.

    Attributes:
        names: Operator names, in the order weights are held.
        weights: Current selection weights.
        scores: Score accumulated in the current segment.
        uses: Selections made in the current segment.
        decay: How much of the old weight survives a segment boundary.
    """

    names: tuple[str, ...]
    weights: list[float] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    uses: list[int] = field(default_factory=list)
    decay: float = 0.8

    def __post_init__(self) -> None:
        count = len(self.names)
        if not self.weights:
            self.weights = [1.0] * count
        self.scores = [0.0] * count
        self.uses = [0] * count

    def choose(self, rng: random.Random) -> int:
        """Pick an operator index, proportionally to weight."""
        total = sum(self.weights)
        if total <= 0.0:
            return rng.randrange(len(self.names))
        threshold = rng.random() * total
        running = 0.0
        for index, weight in enumerate(self.weights):
            running += weight
            if running >= threshold:
                return index
        return len(self.weights) - 1

    def reward(self, index: int, amount: float) -> None:
        """Credit an operator for an outcome."""
        self.scores[index] += amount
        self.uses[index] += 1

    def end_segment(self) -> None:
        """Fold the segment's scores into the weights and reset."""
        for index in range(len(self.names)):
            if self.uses[index]:
                average = self.scores[index] / self.uses[index]
                self.weights[index] = self.decay * self.weights[index] + (1 - self.decay) * average
            self.scores[index] = 0.0
            self.uses[index] = 0

    def report(self) -> str:
        """Weights as a one-line string, for run logs."""
        return " ".join(
            f"{name}={weight:.3f}" for name, weight in zip(self.names, self.weights, strict=True)
        )


@dataclass(frozen=True, slots=True)
class AdaptiveLNS:
    """Ruin and recreate with adaptive operator selection and annealing acceptance.

    Attributes:
        iterations: How many ruin-recreate rounds to run.
        min_removal: Fewest requests to tear out per round.
        max_removal_fraction: Most to tear out, as a fraction of those assigned.
        seed: Random seed. Fixed by default, because a result nobody can re-run
            is not a result.
        start_temperature_fraction: Initial annealing temperature, as a fraction
            of the starting solution's cost.
        cooling: Per-iteration temperature multiplier.
        segment: Iterations between operator-weight updates.
        reward_new_best: Credit for finding a new global best.
        reward_improved: Credit for improving on the incumbent.
        reward_accepted: Credit for a worse solution the acceptance rule took.
        checkpoint_every: Iterations between recorded checkpoints.
    """

    iterations: int = 1000
    min_removal: int = 2
    max_removal_fraction: float = 0.35
    seed: int = 20260920
    start_temperature_fraction: float = 0.05
    cooling: float = 0.9975
    segment: int = 100
    reward_new_best: float = 10.0
    reward_improved: float = 4.0
    reward_accepted: float = 1.0
    checkpoint_every: int = 50

    @property
    def name(self) -> str:
        """Identifier for the results table."""
        return "adaptive_lns"

    def solve(self, study: Study, initial: Solution | None = None) -> RunResult:
        """Run ruin-and-recreate for :attr:`iterations` rounds."""
        recorder = RunRecorder()
        rng = random.Random(self.seed)
        repair = RegretInsertion()

        current = initial if initial is not None else repair.solve(study).solution
        best = current
        current_cost = study.penalised_cost(current)
        best_cost = current_cost

        temperature = max(1e-9, abs(current_cost) * self.start_temperature_fraction)
        operators = OperatorScores(tuple(name for name, _ in DEFAULT_OPERATORS))

        for iteration in range(1, self.iterations + 1):
            recorder.tick()
            index = operators.choose(rng)
            _, ruin = DEFAULT_OPERATORS[index]

            assigned = len(current.assigned_requests())
            if assigned == 0:
                break
            ceiling = max(self.min_removal, int(assigned * self.max_removal_fraction))
            count = rng.randint(self.min_removal, max(self.min_removal, ceiling))

            ruined = ruin(study, current, count, rng).apply(current)
            candidate = repair.solve(study, ruined).solution
            candidate_cost = study.penalised_cost(candidate)
            recorder.count_evaluations(1)

            if candidate_cost < best_cost:
                best, best_cost = candidate, candidate_cost
                current, current_cost = candidate, candidate_cost
                operators.reward(index, self.reward_new_best)
            elif candidate_cost < current_cost:
                current, current_cost = candidate, candidate_cost
                operators.reward(index, self.reward_improved)
            elif _accept(candidate_cost, current_cost, temperature, rng):
                current, current_cost = candidate, candidate_cost
                operators.reward(index, self.reward_accepted)
            else:
                operators.reward(index, 0.0)
                # Nothing to undo: `current` was never touched.

            temperature *= self.cooling
            if iteration % self.segment == 0:
                operators.end_segment()
            if iteration % self.checkpoint_every == 0:
                recorder.checkpoint(study, best)

        return recorder.finish(
            study,
            best,
            {
                "operator_weights": operators.report(),
                "final_temperature": f"{temperature:.6g}",
                "seed": str(self.seed),
            },
        )


def _accept(candidate: float, current: float, temperature: float, rng: random.Random) -> bool:
    """Metropolis acceptance of a worse candidate."""
    if temperature <= 0.0:
        return False
    exponent = (current - candidate) / temperature
    if exponent < -700.0:  # guard against underflow in exp
        return False
    return rng.random() < math.exp(exponent)
