"""Abstract base class for simulation dataset → Octave octree loaders.

To add a new code, subclass OctaveInterface, implement load(), and register
the key in get_interface() at the bottom of this file.
"""
from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Shared field-layout constants (re-exported for callers)
# ─────────────────────────────────────────────────────────────────────────────

SHOCKFIND_FIELDS_MHD   = ["density", "pressure", "vx", "vy", "vz", "bx", "by", "bz"]
SHOCKFIND_FIELDS_HYDRO = ["density", "pressure", "vx", "vy", "vz"]
SHOCKFIND_FIELDS = SHOCKFIND_FIELDS_MHD   # backward-compatible alias


# ─────────────────────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OctreeData:
    """Normalised data ready for octree deposition."""
    positions:    np.ndarray                  # (N, 3) float64, [0, 1]^3
    attrs:        np.ndarray                  # (N, K) float64
    field_names:  list[str]                   # length K
    hsml:         Optional[np.ndarray] = None # (N,) — None for AMR/grid codes
    particle_ids: Optional[np.ndarray] = None # (N,) — None if unavailable


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class OctaveInterface(ABC):
    """
    Common API for loading any simulation code into an Octave octree.

    Every concrete subclass must implement load() and return an OctreeData.
    All other parameters are shared across all interfaces so that build_octree.py
    and the shockfind pipeline can treat every code identically.

    Parameters — common to all codes
    ─────────────────────────────────
    data_path   : snapshot file or dataset directory
    box         : sub-region [xmin,xmax, ymin,ymax, zmin,zmax] in ``box_units``
    box_units   : 'code' (simulation units) | 'norm' (normalised [0,1]^3)
    hydro_only  : skip magnetic field loading; use sonic-only shock classification
    max_depth   : finest octree level  (cell size = 1/2^max_depth)
    min_depth   : coarsest pre-built level
    max_members : max data points per leaf before splitting
    max_nodes   : hard cap on total octree nodes

    Parameters — particle/SPH codes only (ignored for AMR grid codes)
    ──────────────────────────────────────────────────────────────────
    part_type   : particle type to load  (0 = gas)
    hsml_factor : multiplicative scale on stored/computed smoothing lengths
    hsml_geom   : influence region: 'sphere' | 'cube'
    split_mass  : divide mass among intersecting leaves (mass-conserving SPH)
    """

    def __init__(
        self,
        data_path: str,
        *,
        # sub-box
        box=None,
        box_units:   str   = "code",
        # physics
        hydro_only:  bool  = False,
        # particle-code options (ignored for AMR)
        part_type:   int   = 0,
        hsml_factor: float = 1.0,
        hsml_geom:   str   = "sphere",
        split_mass:  bool  = False,
        # octree build
        max_depth:   int   = 12,
        min_depth:   int   = 2,
        max_members: int   = 1,
        max_nodes:   int   = 100_000_000,
    ):
        self.data_path   = data_path
        self.box         = box
        self.box_units   = box_units
        self.hydro_only  = hydro_only
        self.part_type   = part_type
        self.hsml_factor = hsml_factor
        self.hsml_geom   = hsml_geom
        self.split_mass  = split_mass
        self.max_depth   = max_depth
        self.min_depth   = min_depth
        self.max_members = max_members
        self.max_nodes   = max_nodes

    # ── abstract ─────────────────────────────────────────────────────────────

    @abstractmethod
    def load(self) -> OctreeData:
        """
        Load the simulation snapshot into normalised arrays.

        Positions must be in [0, 1]^3.
        Set hsml=None for AMR / uniform-grid codes.
        """

    # ── overridable properties ─────────────────────────────────────────────────

    @property
    def uses_hsml(self) -> bool:
        """True for SPH / moving-mesh codes; False for AMR / grid codes."""
        return False

    @property
    def has_bfield(self) -> bool:
        """True when magnetic field (bx/by/bz) is available in the loaded data."""
        return not self.hydro_only

    @property
    def shockfind_fields(self) -> list[str]:
        """Ordered field names expected by ShockFind (subset of full field list)."""
        return SHOCKFIND_FIELDS_HYDRO if self.hydro_only else SHOCKFIND_FIELDS_MHD

    @property
    def octree_params(self) -> dict:
        return {
            "max_depth":   self.max_depth,
            "min_depth":   self.min_depth,
            "max_members": self.max_members,
            "max_nodes":   self.max_nodes,
        }

    def register_fields(self, tree) -> None:
        """Register per-field reducers on an octree3d.Octree3D.

        Default is a no-op; SPH codes override this to set mass-weighted means.
        """

    # ── sub-box helper ─────────────────────────────────────────────────────────

    @staticmethod
    def parse_box(
        *,
        box=None,
        center=None,
        box_side: float | None = None,
    ) -> tuple[list | None, str]:
        """
        Resolve sub-box arguments into (box_6vals, 'code').

        Two modes (mutually exclusive):
          1. box=[xmin,xmax, ymin,ymax, zmin,zmax] — explicit corners in code units
          2. center=[cx,cy,cz] + box_side=S        — centred cube of side S
        Returns (None, 'code') when no box is specified.
        """
        if box is not None:
            return list(box), "code"
        if center is not None or box_side is not None:
            if center is None or box_side is None:
                raise ValueError("center and box_side must both be given together")
            cx, cy, cz = center
            half = box_side / 2.0
            return [cx - half, cx + half,
                    cy - half, cy + half,
                    cz - half, cz + half], "code"
        return None, "code"


# ─────────────────────────────────────────────────────────────────────────────
# Shared octave3d import helper
# ─────────────────────────────────────────────────────────────────────────────

def _import_octave():
    """Import the octree3d (Octave) Python extension."""
    try:
        import octree3d
        return octree3d
    except ImportError:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", ".."),
        os.path.expanduser("~/codes/Octave"),
        os.path.expanduser("~/codes/Octave/build"),
    ]
    for path in candidates:
        path = os.path.abspath(path)
        if os.path.isdir(path):
            sys.path.insert(0, path)
            try:
                import octree3d
                return octree3d
            except ImportError:
                sys.path.pop(0)
    raise ImportError(
        "octree3d not found. Build Octave with:\n"
        "  cd ~/codes/Octave && pip install -e . --no-build-isolation"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Interface factory
# ─────────────────────────────────────────────────────────────────────────────

def get_interface(kind: str, data_path: str, **kwargs) -> OctaveInterface:
    """
    Instantiate a dataset loader by name.

    Parameters
    ----------
    kind      : 'arepo' | 'ramses'  (case-insensitive; extend _REGISTRY for new codes)
    data_path : path to the snapshot file or dataset directory
    **kwargs  : forwarded verbatim to OctaveInterface.__init__
    """
    key = kind.strip().lower()
    if key == "arepo":
        from .Arepo import ArepoInterface
        return ArepoInterface(data_path, **kwargs)
    elif key == "ramses":
        from .Ramses import RamsesInterface
        return RamsesInterface(data_path, **kwargs)
    else:
        raise ValueError(
            f"Unknown interface '{kind}'. Available: arepo, ramses\n"
            "Add new codes by subclassing OctaveInterface in src/interfaces/ "
            "and registering them in get_interface()."
        )
