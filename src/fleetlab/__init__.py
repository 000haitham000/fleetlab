"""fleetlab -- a framework for simulation-based optimisation of pickup-and-delivery problems.

What is being moved may be people or goods. The core never asks: a
:class:`~fleetlab.domain.loadable.Loadable` is whatever occupies capacity, and
capacity is a vector over named dimensions, so seats, wheelchair bays, mass and
volume are all just dimensions. Dial-a-ride is the configuration where
per-request onboard-time caps are active; goods distribution is the one where
they are not.

The framework has two lanes over one problem statement:

* a **search lane**, where a schedule is known and is evaluated;
* a **mathematical-programming lane**, where the schedule is what a solver
  chooses.

Every constraint and every objective term states itself in both, so the bound
and the heuristic are provably answering the same question.

Start with :class:`fleetlab.study.Study`.
"""

from fleetlab.study import Study

__version__ = "0.1.0"

__all__ = ["Study", "__version__"]
