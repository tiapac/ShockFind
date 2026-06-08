"""Dataset → Octave octree interface registry.

Quick start:
    from src.interfaces import get_interface
    iface = get_interface("arepo", "snapshot_082.hdf5", max_depth=12)
    data  = iface.load()      # → OctreeData
"""
from .interface import (
    OctaveInterface,
    OctreeData,
    get_interface,
    SHOCKFIND_FIELDS_MHD,
    SHOCKFIND_FIELDS_HYDRO,
    SHOCKFIND_FIELDS,
    _import_octave,
)

__all__ = [
    "OctaveInterface",
    "OctreeData",
    "get_interface",
    "SHOCKFIND_FIELDS_MHD",
    "SHOCKFIND_FIELDS_HYDRO",
    "SHOCKFIND_FIELDS",
    "_import_octave",
]
