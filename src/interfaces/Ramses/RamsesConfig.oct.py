"""RAMSES interface configuration for the Octave octree pipeline.

Loads RAMSES AMR simulation data via yt.  Supports both MHD and hydro-only runs.
Optional fields (tracers, passive scalars) are loaded silently if present.

Edit the DEFAULT_* constants below to change global defaults for your setup.
All values can be overridden per-call via keyword arguments.

Usage
-----
    from src.interfaces import get_interface
    iface = get_interface("ramses", "/path/to/output_00021", hydro_only=False)
    data  = iface.load()
"""
from __future__ import annotations

import os
import sys
import glob
import numpy as np

try:
    from ..interface import (
        OctaveInterface, OctreeData,
        SHOCKFIND_FIELDS_MHD, SHOCKFIND_FIELDS_HYDRO,
    )
except ImportError:
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_here, "..", ".."))
    from interfaces.interface import (
        OctaveInterface, OctreeData,
        SHOCKFIND_FIELDS_MHD, SHOCKFIND_FIELDS_HYDRO,
    )


# ─────────────────────────────────────────────────────────────────────────────
# User-editable defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MAX_DEPTH   = 14
DEFAULT_MIN_DEPTH   = 3
DEFAULT_MAX_MEMBERS = 1     # mirrors RAMSES AMR: each leaf holds one cell
DEFAULT_MAX_NODES   = 100_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Field maps  (ShockFind name → list of (yt_field_type, yt_field_name) to try)
# ─────────────────────────────────────────────────────────────────────────────

_FIELD_MAP: dict[str, list[tuple[str, str]]] = {
    "density":  [("gas", "density")],
    "pressure": [("gas", "pressure")],
    "vx":       [("gas", "velocity_x")],
    "vy":       [("gas", "velocity_y")],
    "vz":       [("gas", "velocity_z")],
    "bx":       [("gas", "magnetic_field_x")],
    "by":       [("gas", "magnetic_field_y")],
    "bz":       [("gas", "magnetic_field_z")],
}

# Optional fields — silently skipped if absent in the dataset
_OPTIONAL_FIELDS: dict[str, list[tuple[str, str]]] = {
    "hydro_scalar_00": [("ramses", "hydro_scalar_00")],
}


# ─────────────────────────────────────────────────────────────────────────────
# Concrete interface
# ─────────────────────────────────────────────────────────────────────────────

class RamsesInterface(OctaveInterface):
    """Load a RAMSES AMR snapshot via yt.

    RAMSES is a grid code — no HSML.  The octree is built by inserting one
    point per cell (max_members=1 mirrors the AMR hierarchy exactly).
    """

    def __init__(self, data_path: str, **kwargs):
        kwargs.setdefault("max_depth",   DEFAULT_MAX_DEPTH)
        kwargs.setdefault("min_depth",   DEFAULT_MIN_DEPTH)
        kwargs.setdefault("max_members", DEFAULT_MAX_MEMBERS)
        kwargs.setdefault("max_nodes",   DEFAULT_MAX_NODES)
        super().__init__(data_path, **kwargs)

    @property
    def uses_hsml(self) -> bool:
        return False

    # ── data loading ─────────────────────────────────────────────────────────

    def load(self) -> OctreeData:
        """Load RAMSES cells via yt → normalised positions and field arrays."""
        try:
            import yt
        except ImportError:
            raise RuntimeError(
                "yt is required for RAMSES loading: pip install yt")

        info_path = self.data_path
        if os.path.isfile(info_path):
            info = info_path
        else:
            candidates = sorted(glob.glob(os.path.join(info_path, "info_*.txt")))
            if not candidates:
                raise FileNotFoundError(f"No info_*.txt under {info_path}")
            info = candidates[-1]

        ds = yt.load(info)
        ad = ds.all_data()

        le = ds.domain_left_edge.to("code_length").v
        w  = ds.domain_width.to("code_length").v

        x = ad[("gas", "x")].to("code_length").v
        y = ad[("gas", "y")].to("code_length").v
        z = ad[("gas", "z")].to("code_length").v

        positions = np.column_stack([
            np.clip((x - le[0]) / w[0], 0.0, 1.0),
            np.clip((y - le[1]) / w[1], 0.0, 1.0),
            np.clip((z - le[2]) / w[2], 0.0, 1.0),
        ])

        # Sub-box filter
        mask = None
        if self.box is not None:
            bx = np.array(self.box, dtype=float)
            if self.box_units == "code":
                bx[:2]  = (bx[:2]  - le[0]) / w[0]
                bx[2:4] = (bx[2:4] - le[1]) / w[1]
                bx[4:]  = (bx[4:]  - le[2]) / w[2]
            mask = (
                (positions[:, 0] >= bx[0]) & (positions[:, 0] <= bx[1]) &
                (positions[:, 1] >= bx[2]) & (positions[:, 1] <= bx[3]) &
                (positions[:, 2] >= bx[4]) & (positions[:, 2] <= bx[5])
            )
            positions = positions[mask]

        # Core ShockFind fields
        base_fields = SHOCKFIND_FIELDS_HYDRO if self.hydro_only else SHOCKFIND_FIELDS_MHD
        cols: list[np.ndarray] = []

        for name in base_fields:
            val = None
            for yt_key in _FIELD_MAP[name]:
                if yt_key in ds.derived_field_list:
                    arr = np.asarray(ad[yt_key], dtype=float)
                    if mask is not None:
                        arr = arr[mask]
                    val = arr
                    break
            if val is None:
                raise RuntimeError(
                    f"Field '{name}' not found in RAMSES dataset.\n"
                    f"Available: {ds.derived_field_list}")
            cols.append(val)

        # Optional fields
        optional_loaded: list[str] = []
        for opt_name, yt_keys in _OPTIONAL_FIELDS.items():
            for yt_key in yt_keys:
                if yt_key in ds.derived_field_list:
                    arr = np.asarray(ad[yt_key], dtype=float)
                    if mask is not None:
                        arr = arr[mask]
                    cols.append(arr)
                    optional_loaded.append(opt_name)
                    break
            else:
                pass   # optional → silent skip

        all_field_names = base_fields + optional_loaded
        attrs = np.column_stack(cols)

        mode_label = "hydro" if self.hydro_only else "MHD"
        extra = f" + {optional_loaded}" if optional_loaded else ""
        print(
            f"Loaded {positions.shape[0]:,} RAMSES cells  "
            f"({len(base_fields)} {mode_label} fields{extra})"
        )

        return OctreeData(
            positions=positions.astype(np.float64, copy=False),
            attrs=attrs.astype(np.float64, copy=False),
            field_names=all_field_names,
            hsml=None,
            particle_ids=None,
        )
