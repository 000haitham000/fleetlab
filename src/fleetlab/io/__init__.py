"""Reading, writing and generating instances."""

from fleetlab.io.generate import MIXED_SPACE, mixed_instance, tiny_instance
from fleetlab.io.instance import FORMAT_VERSION, from_dict, load, save, to_dict

__all__ = [
    "FORMAT_VERSION",
    "MIXED_SPACE",
    "from_dict",
    "load",
    "mixed_instance",
    "save",
    "tiny_instance",
    "to_dict",
]
