"""Arepo interface configuration for the Octave octree pipeline.

Handles both SPH snapshots (SmoothingLength present) and moving-mesh / Voronoi
snapshots (HSML computed from mass and density).  Magnetic field is optional
(absent → hydro_only automatically set, or set hydro_only=True explicitly).

Edit the DEFAULT_* constants below to change global defaults for your setup.
All values can be overridden per-call via keyword arguments.

Usage
-----
    from src.interfaces import get_interface
    iface = get_interface("arepo", "snapshot_082.hdf5", max_depth=12)
    data  = iface.load()
"""
from __future__ import annotations

import os
import sys
import numpy as np
from typing import Optional

# Re-import parent so this file can also be run standalone
try:
    from ..interface import OctaveInterface, OctreeData
except ImportError:
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_here, "..", ".."))
    from interfaces.interface import OctaveInterface, OctreeData


# ─────────────────────────────────────────────────────────────────────────────
# User-editable defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MAX_DEPTH   = 12
DEFAULT_MIN_DEPTH   = 2
DEFAULT_MAX_MEMBERS = 8       # SPH / moving-mesh: 8 recommended
DEFAULT_MAX_NODES   = 200_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Field definitions (HDF5 key → output name → reducer)
# ─────────────────────────────────────────────────────────────────────────────

_MANDATORY_SCALAR = [
    ("Masses",         "mass",     "sum"),
    ("Density",        "density",  "wmean"),
    ("InternalEnergy", "u",        "wmean"),
    ("Pressure",       "pressure", "wmean"),
]

_MANDATORY_VECTOR = ("Velocities", ("vx", "vy", "vz"), "wmean")

_OPTIONAL_SCALAR = [
    ("ElectronAbundance",         "ne",           "wmean"),
    ("NeutralHydrogenAbundance",  "nh",           "wmean"),
    ("MolecularHFrac",            "mol_h_frac",   "wmean"),
    ("GFM_Metallicity",           "metallicity",  "wmean"),
    ("StarFormationRate",         "sfr",          "wmean"),
    ("GFM_CoolingRate",           "cooling_rate", "wmean"),
    ("VirialParameter",           "virial",       "wmean"),
    ("GasRadCoolShutoffTime",     "cool_shutoff", "wmean"),
    ("Potential",                 "potential",    "wmean"),
]

_OPTIONAL_VECTOR = [
    ("MagneticField", ("bx", "by", "bz"), "wmean"),
]

_OPTIONAL_MULTICOL = [
    ("PassiveScalars", "passive_scalar", "wmean"),
    ("GFM_Metals",     "metal_frac",     "wmean"),
]

# field_name → (reducer, weight_field | None)
_REDUCERS: dict[str, tuple[str, str | None]] = {
    "mass":         ("sum",   None),
    "density":      ("wmean", "mass"),
    "u":            ("wmean", "mass"),
    "pressure":     ("wmean", "mass"),
    "vx":           ("wmean", "mass"),
    "vy":           ("wmean", "mass"),
    "vz":           ("wmean", "mass"),
    "bx":           ("wmean", "mass"),
    "by":           ("wmean", "mass"),
    "bz":           ("wmean", "mass"),
    "ne":           ("wmean", "mass"),
    "nh":           ("wmean", "mass"),
    "mol_h_frac":   ("wmean", "mass"),
    "metallicity":  ("wmean", "mass"),
    "sfr":          ("wmean", "mass"),
    "cooling_rate": ("wmean", "mass"),
    "virial":       ("wmean", "mass"),
    "cool_shutoff": ("wmean", "mass"),
    "potential":    ("wmean", "mass"),
    "temperature":  ("wmean", "mass"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Concrete interface
# ─────────────────────────────────────────────────────────────────────────────

class ArepoInterface(OctaveInterface):
    """Load an Arepo HDF5 snapshot into normalised arrays for octree deposition.

    SPH snapshots use the stored SmoothingLength.
    Moving-mesh / Voronoi snapshots (no SmoothingLength) compute:
        r_eff = hsml_factor * (3*m / (4*pi*rho))^(1/3)
    """

    def __init__(self, data_path: str, **kwargs):
        # Arepo-specific defaults — override base defaults before passing up
        kwargs.setdefault("max_depth",   DEFAULT_MAX_DEPTH)
        kwargs.setdefault("min_depth",   DEFAULT_MIN_DEPTH)
        kwargs.setdefault("max_members", DEFAULT_MAX_MEMBERS)
        kwargs.setdefault("max_nodes",   DEFAULT_MAX_NODES)
        super().__init__(data_path, **kwargs)

    @property
    def uses_hsml(self) -> bool:
        return True

    # ── field reducer registration ──────────────────────────────────────────

    def register_fields(self, tree) -> None:
        """Register mass-weighted reducers on an octree3d.Octree3D."""
        # Called after load(); self._field_names is set by load()
        field_names = getattr(self, "_field_names", [])
        mass_col = field_names.index("mass") if "mass" in field_names else -1
        for i, name in enumerate(field_names):
            entry = _REDUCERS.get(name, ("wmean", "mass"))
            reducer, weight_name = entry
            weight_col = mass_col if (weight_name == "mass" and mass_col >= 0) else None
            if weight_col is not None:
                tree.register_field(name, i, reducer, weight_col)
            else:
                tree.register_field(name, i, reducer)

    # ── data loading ─────────────────────────────────────────────────────────

    def load(self) -> OctreeData:
        """Load Arepo HDF5 snapshot → normalised positions, field arrays, HSML."""
        try:
            import h5py
        except ImportError:
            raise RuntimeError("h5py is required: pip install h5py")

        snap_path = self.data_path
        part_type = self.part_type
        box       = self.box
        box_units = self.box_units
        pkey      = f"PartType{part_type}"

        with h5py.File(snap_path, "r") as f:
            header       = dict(f["Header"].attrs)
            boxsize      = float(header.get("BoxSize", 0.0))
            unit_vel_cgs = float(header.get("UnitVelocity_in_cm_per_s", 0.0))
            if unit_vel_cgs == 0.0 and "Parameters" in f:
                unit_vel_cgs = float(
                    dict(f["Parameters"].attrs).get("UnitVelocity_in_cm_per_s", 0.0))
            if unit_vel_cgs == 0.0:
                unit_vel_cgs = 1.0e5   # fallback: 1 km/s

            coords       = f[pkey]["Coordinates"][:]
            keys_in_snap = set(f[pkey].keys())

            # Reference scale
            if boxsize > 0.0:
                origin = np.zeros(3)
                L_full = float(boxsize)
            else:
                origin = coords.min(axis=0)
                L_full = float((coords.max(axis=0) - origin).max()) or 1.0

            # Sub-box filter
            mask = None
            if box is not None:
                bx = np.array(box, dtype=float).reshape(3, 2)
                if box_units == "norm":
                    bx = bx * L_full + origin[:, np.newaxis]
                mask = (
                    (coords[:, 0] >= bx[0, 0]) & (coords[:, 0] <= bx[0, 1]) &
                    (coords[:, 1] >= bx[1, 0]) & (coords[:, 1] <= bx[1, 1]) &
                    (coords[:, 2] >= bx[2, 0]) & (coords[:, 2] <= bx[2, 1])
                )
                sub_origin = bx[:, 0]
                L = float((bx[:, 1] - bx[:, 0]).max()) or 1.0
                positions = ((coords[mask] - sub_origin) / L).astype(np.float64, copy=False)
            else:
                L = L_full
                positions = ((coords - origin) / L).astype(np.float64, copy=False)

            def _load(key) -> Optional[np.ndarray]:
                if key not in keys_in_snap:
                    return None
                arr = f[pkey][key][:]
                if mask is not None:
                    arr = arr[mask]
                return arr.astype(np.float64, copy=False)

            # HSML
            if "SmoothingLength" in keys_in_snap:
                hsml = _load("SmoothingLength") / L
                hsml_source = "SmoothingLength"
            else:
                mass_raw = _load("Masses")
                dens_raw = _load("Density")
                if mass_raw is None or dens_raw is None:
                    raise RuntimeError(
                        "No SmoothingLength and no Masses/Density to compute HSML from.")
                vol  = mass_raw / np.maximum(dens_raw, 1e-300)
                hsml = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0) / L
                hsml_source = "computed (m/rho)^(1/3)"
            if self.hsml_factor != 1.0:
                hsml *= self.hsml_factor

            particle_ids = _load("ParticleIDs")

            # Collect field columns
            cols: list[np.ndarray] = []
            field_names: list[str] = []
            _u_arr:  Optional[np.ndarray] = None
            _ne_arr: Optional[np.ndarray] = None

            # Mandatory scalars
            for hdf_key, name, _ in _MANDATORY_SCALAR:
                arr = _load(hdf_key)
                if arr is None:
                    raise RuntimeError(
                        f"Mandatory field '{hdf_key}' not found in {pkey}.\n"
                        f"Available: {sorted(keys_in_snap)}")
                cols.append(arr)
                field_names.append(name)
                if name == "u":
                    _u_arr = arr

            # Mandatory vector
            hdf_key, comps, _ = _MANDATORY_VECTOR
            arr = _load(hdf_key)
            if arr is None:
                arr = np.zeros((len(positions), 3), dtype=np.float64)
            for i, comp in enumerate(comps):
                cols.append(arr[:, i])
                field_names.append(comp)

            # Optional scalars
            for hdf_key, name, _ in _OPTIONAL_SCALAR:
                arr = _load(hdf_key)
                if arr is not None and arr.ndim == 1:
                    cols.append(arr)
                    field_names.append(name)
                    if name == "ne":
                        _ne_arr = arr

            # Optional vectors — skip B if hydro_only
            for hdf_key, comps, _ in _OPTIONAL_VECTOR:
                if self.hydro_only and hdf_key == "MagneticField":
                    continue
                arr = _load(hdf_key)
                if arr is not None and arr.ndim == 2:
                    for i, comp in enumerate(comps):
                        cols.append(arr[:, i])
                        field_names.append(comp)

            # Optional multi-column scalars
            for hdf_key, prefix, _ in _OPTIONAL_MULTICOL:
                arr = _load(hdf_key)
                if arr is not None and arr.ndim == 2:
                    for i in range(arr.shape[1]):
                        cols.append(arr[:, i])
                        field_names.append(f"{prefix}_{i}")

        # Temperature (outside h5py context)
        if _u_arr is not None:
            _GAMMA  = 5.0 / 3.0
            _MP_CGS = 1.6726e-24
            _KB_CGS = 1.3806e-16
            _XH     = 0.76
            u_cgs = _u_arr * unit_vel_cgs ** 2
            if _ne_arr is not None:
                mu = 4.0 / (1.0 + 3.0 * _XH + 4.0 * _XH * _ne_arr)
            else:
                mu = 0.6
            temperature = (_GAMMA - 1.0) * u_cgs * mu * _MP_CGS / _KB_CGS
            cols.append(temperature.astype(np.float64, copy=False))
            field_names.append("temperature")

        attrs     = np.column_stack(cols).astype(np.float64, copy=False)
        positions = positions.astype(np.float64, copy=False)
        hsml      = np.clip(hsml, 0.0, None).astype(np.float64, copy=False)

        scale_label = f"sub-box side={L:.4g}" if box is not None else f"BoxSize={L:.4g}"
        mode_label  = "hydro" if self.hydro_only else "MHD+hydro"
        print(
            f"Loaded {len(positions):,} Arepo PartType{part_type} particles"
            f"  ({scale_label} code units, [0,1]^3)  [{mode_label}]"
        )
        print(
            f"  HSML: {hsml_source}"
            f"  min={hsml.min():.3e}  median={np.median(hsml):.3e}  max={hsml.max():.3e}"
        )
        print(f"  Fields ({len(field_names)}): {', '.join(field_names)}")

        # Cache field names for register_fields()
        self._field_names = field_names
        return OctreeData(
            positions=positions,
            attrs=attrs,
            field_names=field_names,
            hsml=hsml,
            particle_ids=particle_ids,
        )
