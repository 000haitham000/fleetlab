"""Identifier types.

These are ``NewType`` aliases over ``str``. They cost nothing at runtime and let
mypy catch the class of bug where a stop id is passed where a request id was
meant -- which is easy to do once a schedule is a tuple of small value objects.
"""

from __future__ import annotations

from typing import NewType

StopId = NewType("StopId", str)
RequestId = NewType("RequestId", str)
LoadableId = NewType("LoadableId", str)
VehicleId = NewType("VehicleId", str)

__all__ = ["LoadableId", "RequestId", "StopId", "VehicleId"]
