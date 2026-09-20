"""Compare a construction heuristic, a metaheuristic and an exact bound.

Run with::

    python examples/run_study.py

This is the shape a study's own experiment script takes: build an instance once,
hand the same :class:`~fleetlab.study.Study` to every algorithm, and print the
per-term cost breakdown rather than a single number -- so the table says *why*
one approach beat another.
"""

from __future__ import annotations

from fleetlab.io.generate import mixed_instance, tiny_instance
from fleetlab.mathprog import build_model, solution_to_assignment
from fleetlab.search import AdaptiveLNS, CheapestInsertion, RegretInsertion
from fleetlab.simulation import Simulator
from fleetlab.study import Study


def compare_heuristics() -> None:
    """Run two construction heuristics and a metaheuristic on one instance."""
    problem = mixed_instance(passengers=8, wheelchair_users=2, parcels=6, vehicles=3)
    print(f"\n=== {problem} ===\n")

    baseline = None
    for algorithm in (CheapestInsertion(), RegretInsertion(k=3)):
        study = Study(problem)
        result = algorithm.solve(study)
        print(f"--- {algorithm.name}")
        print(result.describe())
        print()
        if algorithm.name.startswith("regret"):
            baseline = result.solution

    study = Study(problem)
    improved = AdaptiveLNS(iterations=600).solve(study, baseline)
    print("--- adaptive_lns")
    print(improved.describe())
    print()


def exact_bound() -> None:
    """Build a mathematical program and, if a solver is installed, solve it."""
    problem = tiny_instance()
    study = Study(problem)
    heuristic = RegretInsertion().solve(study)

    report = build_model(study)
    print(f"\n=== exact bound for {problem} ===\n")
    print(report.describe())
    print(f"\nheuristic cost: {heuristic.cost:.4f}")

    try:
        from fleetlab.mathprog.adapters import solve_with_pulp
    except ImportError:
        print("\nNo solver installed. Write the model with:")
        print("    from fleetlab.mathprog.adapters import write_lp")
        print("    write_lp(report.model, 'instance.lp')")
        return

    warm = solution_to_assignment(study, report.variables, heuristic.solution)
    solved = solve_with_pulp(report.model, time_limit=120, warm_start=warm)
    print(f"solver status:  {solved.status}")
    if solved.objective is not None:
        print(f"bound:          {solved.objective:.4f}")
        gap = (heuristic.cost - solved.objective) / max(abs(solved.objective), 1e-9)
        print(f"heuristic gap:  {gap:.2%}")


def dynamic_day() -> None:
    """Simulate a day in which requests arrive over time."""
    problem = mixed_instance(passengers=6, wheelchair_users=1, parcels=4, vehicles=3)
    study = Study(problem)
    result = Simulator(cadence=30.0).run(study, RegretInsertion())
    print("\n=== simulated day ===\n")
    print(result.describe())


if __name__ == "__main__":
    compare_heuristics()
    exact_bound()
    dynamic_day()
