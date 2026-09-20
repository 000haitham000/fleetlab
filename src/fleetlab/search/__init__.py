"""Search: the heuristic lane.

Every algorithm implements one method -- ``solve(study, initial) -> RunResult`` --
so that a comparison table is comparing like with like. What ships here is a
baseline and a worked reference, not a finished competitor; a study's own
algorithms are the point.
"""

from fleetlab.search.base import Algorithm, Checkpoint, RunRecorder, RunResult
from fleetlab.search.greedy import CheapestInsertion, RegretInsertion
from fleetlab.search.lns import AdaptiveLNS, OperatorScores

__all__ = [
    "AdaptiveLNS",
    "Algorithm",
    "CheapestInsertion",
    "Checkpoint",
    "OperatorScores",
    "RegretInsertion",
    "RunRecorder",
    "RunResult",
]
