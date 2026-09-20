"""Travel matrices.

The protocol in :mod:`fleetlab.od.base` requires every matrix to declare its
time dependence, so that the mathematical-programming lane can tell the
difference between an instance it can represent exactly and one it cannot.
"""

from fleetlab.od.base import BucketedODMatrix, ODMatrix, TimeDependence, mean_duration
from fleetlab.od.constant import ConstantODMatrix
from fleetlab.od.euclidean import EuclideanODMatrix
from fleetlab.od.piecewise import PiecewiseODMatrix

__all__ = [
    "BucketedODMatrix",
    "ConstantODMatrix",
    "EuclideanODMatrix",
    "ODMatrix",
    "PiecewiseODMatrix",
    "TimeDependence",
    "mean_duration",
]
