"""The mathematical-programming lane.

A study's constraints and objective render themselves into rows and an
expression; this package supplies the variables, the structural rows that make a
set of arcs into a schedule, and the round trip back to a
:class:`~fleetlab.domain.solution.Solution`.

There is no solver dependency. :meth:`~fleetlab.linear.Model.to_lp` writes a
standard LP file that any solver reads; :mod:`fleetlab.mathprog.adapters` is
optional and is the only module that imports one.
"""

from fleetlab.mathprog.formulation import BuildReport, build_model
from fleetlab.mathprog.roundtrip import assignment_to_solution, solution_to_assignment
from fleetlab.mathprog.variables import NodeIndex, PickupDeliveryVars

__all__ = [
    "BuildReport",
    "NodeIndex",
    "PickupDeliveryVars",
    "assignment_to_solution",
    "build_model",
    "solution_to_assignment",
]
