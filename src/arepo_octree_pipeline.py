"""
arepo_octree_pipeline.py — Load an Arepo snapshot and deposit gas particles
into an Octave octree using per-particle effective radius (HSML).

For SPH snapshots: uses the stored SmoothingLength directly.
For moving-mesh (Voronoi) snapshots (e.g. Arepo without SmoothingLength):
  computes effective cell radius from mass and density:
    r_eff = (3 * mass / (4 * pi * density))^(1/3)

Each gas particle is deposited into every octree leaf whose AABB intersects
the particle's smoothing sphere.  Intensive fields (velocity, internal energy,
...) are reduced with a mass-weighted mean; mass itself is summed.

Usage (standalone):
    python src/arepo_octree_pipeline.py snapshot_082.hdf5 --save-octree out.h5

Usage (library):
    from src.arepo_octree_pipeline import load_arepo, build_arepo_octree
    positions, hsml, attrs, names = load_arepo("snapshot_082.hdf5")
    tree = build_arepo_octree(positions, hsml, attrs, names, max_depth=12)
    tree.write_all_hdf5("out.h5", names)
"""

from __future__ import annotations

import os
import sys
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Octave bindings import
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
        here,
        os.path.join(here, "build"),
        os.path.expanduser("~/codes/Octave"),
        os.path.expanduser("~/codes/Octave/build"),
    ]
    for path in candidates:
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
# Field definitions
# ─────────────────────────────────────────────────────────────────────────────

# Mandatory scalar fields  — (hdf5_key, output_name, reducer)
_MANDATORY_SCALAR = [
    ("Masses",         "mass",    "sum"),
    ("Density",        "density", "wmean"),
    ("InternalEnergy", "u",       "wmean"),
    ("Pressure",       "pressure","wmean"),
]

# Mandatory vector field
_MANDATORY_VECTOR = ("Velocities", ("vx", "vy", "vz"), "wmean")

# Optional scalar fields — silently skipped if absent
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

# Optional vector fields — each column stored as separate scalar
_OPTIONAL_VECTOR = [
    ("MagneticField", ("bx", "by", "bz"), "wmean"),
]

# Optional multi-column scalar fields — stored as separate scalars
_OPTIONAL_MULTICOL = [
    ("PassiveScalars", "passive_scalar", "wmean"),   # expands to passive_scalar_0, _1, ...
    ("GFM_Metals",     "metal_frac",     "wmean"),   # expands to metal_frac_0 .. _9
]

# Reducer for any field not in the above (fallback)
_DEFAULT_REDUCER = "wmean"

# Maps output field name prefix → reducer
_REDUCERS: dict[str, tuple[str, str | None]] = {
    "mass":           ("sum",   None),
    "density":        ("wmean", "mass"),
    "u":              ("wmean", "mass"),
    "pressure":       ("wmean", "mass"),
    "vx":             ("wmean", "mass"),
    "vy":             ("wmean", "mass"),
    "vz":             ("wmean", "mass"),
    "bx":             ("wmean", "mass"),
    "by":             ("wmean", "mass"),
    "bz":             ("wmean", "mass"),
    "ne":             ("wmean", "mass"),
    "nh":             ("wmean", "mass"),
    "mol_h_frac":     ("wmean", "mass"),
    "metallicity":    ("wmean", "mass"),
    "sfr":            ("wmean", "mass"),
    "cooling_rate":   ("wmean", "mass"),
    "virial":         ("wmean", "mass"),
    "cool_shutoff":   ("wmean", "mass"),
    "potential":      ("wmean", "mass"),
    "temperature":    ("wmean", "mass"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Sub-box helper
# ─────────────────────────────────────────────────────────────────────────────

def resolve_box(
    *,
    box: list | None = None,
    center: list | None = None,
    box_side: float | None = None,
) -> tuple[list | None, str]:
    """
    Resolve sub-box arguments into (box_6vals, 'code') in code units.

    Two ways:
      1. box=[xmin,xmax, ymin,ymax, zmin,zmax]  — explicit corners
      2. center=[cx,cy,cz] + box_side=S         — centred cube of side S

    All values are in Arepo code units.  Returns (None, 'code') if no box given.
    """
    if box is not None:
        return list(box), "code"

    if center is not None or box_side is not None:
        if center is None or box_side is None:
            raise ValueError("--center and --box-side must both be given")
        cx, cy, cz = center
        half = box_side / 2.0
        return [cx - half, cx + half,
                cy - half, cy + half,
                cz - half, cz + half], "code"

    return None, "code"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_arepo(
    snap_path: str,
    part_type: int = 0,
    box: list | None = None,
    box_units: str = "code",
    hsml_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray | None]:
    """
    Load an Arepo HDF5 snapshot and return arrays ready for octree deposition.

    For moving-mesh runs without SmoothingLength, the effective cell radius is
    computed as  r = hsml_factor * (3*m / (4*pi*rho))^(1/3)  and used as HSML.

    Parameters
    ----------
    snap_path   : path to the HDF5 snapshot file
    part_type   : particle type to load (0 = gas)
    box         : [xmin,xmax,ymin,ymax,zmin,zmax] sub-box filter, or None
    box_units   : 'code' (Arepo units) or 'norm' ([0,1])
    hsml_factor : scale factor applied to the computed/stored HSML (default 1.0)

    Returns
    -------
    positions   : (N, 3) float64, normalised to [0, 1]^3
    hsml        : (N,)   float64, smoothing lengths (same normalisation)
    attrs       : (N, K) float64
    field_names : list[str] of length K
    particle_ids: (N,) uint32 or None if not present
    """
    try:
        import h5py
    except ImportError:
        raise RuntimeError("h5py is required: pip install h5py")

    pkey = f"PartType{part_type}"

    with h5py.File(snap_path, "r") as f:
        header  = dict(f["Header"].attrs)
        boxsize = float(header.get("BoxSize", 0.0))

        # Unit velocity for temperature computation
        unit_vel_cgs = float(header.get("UnitVelocity_in_cm_per_s", 0.0))
        if unit_vel_cgs == 0.0 and "Parameters" in f:
            p = dict(f["Parameters"].attrs)
            unit_vel_cgs = float(p.get("UnitVelocity_in_cm_per_s", 0.0))
        if unit_vel_cgs == 0.0:
            unit_vel_cgs = 1.0e5  # default: 1 km/s in cm/s
        coords  = f[pkey]["Coordinates"][:]          # (N, 3) float64
        keys_in_snap = set(f[pkey].keys())

        # Full-box reference scale (always computed; used when no sub-box is given)
        if boxsize > 0.0:
            origin = np.zeros(3)
            L_full = float(boxsize)
        else:
            origin = coords.min(axis=0)
            L_full = float((coords.max(axis=0) - origin).max())
            if L_full == 0.0:
                L_full = 1.0

        # Sub-box: filter in code units then renormalise so the sub-region fills [0,1]^3.
        # Without this the selected particles sit in a tiny corner of the unit cube and
        # the octree wastes all its resolution on empty space.
        mask = None
        if box is not None:
            bx = np.array(box, dtype=float).reshape(3, 2)  # [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
            if box_units == "norm":
                # convert normalised corners back to code units
                bx = bx * L_full + origin[:, np.newaxis]
            mask = (
                (coords[:, 0] >= bx[0, 0]) & (coords[:, 0] <= bx[0, 1]) &
                (coords[:, 1] >= bx[1, 0]) & (coords[:, 1] <= bx[1, 1]) &
                (coords[:, 2] >= bx[2, 0]) & (coords[:, 2] <= bx[2, 1])
            )
            sub_origin = bx[:, 0]                        # min corner in code units
            L = float((bx[:, 1] - bx[:, 0]).max())      # largest side = new unit length
            if L == 0.0:
                L = 1.0
            positions = ((coords[mask] - sub_origin) / L).astype(np.float64, copy=False)
        else:
            L = L_full
            positions = ((coords - origin) / L).astype(np.float64, copy=False)

        def _load(key) -> np.ndarray | None:
            if key not in keys_in_snap:
                return None
            arr = f[pkey][key][:]
            if mask is not None:
                arr = arr[mask]
            return arr.astype(np.float64, copy=False)

        # HSML: stored or computed
        if "SmoothingLength" in keys_in_snap:
            hsml = _load("SmoothingLength") / L
            hsml_source = "SmoothingLength"
        else:
            mass_raw = _load("Masses")
            dens_raw = _load("Density")
            if mass_raw is None or dens_raw is None:
                raise RuntimeError(
                    "No SmoothingLength and no Masses/Density to compute it from."
                )
            # Effective sphere radius from cell volume: V = m/rho, r = (3V/4pi)^(1/3)
            # Normalise: L^3 cancels since mass is in code units and density too
            vol = mass_raw / np.maximum(dens_raw, 1e-300)
            hsml = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0) / L
            hsml_source = "computed (m/rho)^(1/3)"

        if hsml_factor != 1.0:
            hsml = hsml * hsml_factor

        # Particle IDs
        particle_ids = _load("ParticleIDs")

        # Collect field columns
        cols: list[np.ndarray] = []
        field_names: list[str] = []
        _u_arr:  np.ndarray | None = None   # InternalEnergy — needed for temperature
        _ne_arr: np.ndarray | None = None   # ElectronAbundance — needed for mu(T)

        # Mandatory scalars
        for hdf_key, name, _r in _MANDATORY_SCALAR:
            arr = _load(hdf_key)
            if arr is None:
                raise RuntimeError(
                    f"Mandatory field '{hdf_key}' not found in {pkey} of {snap_path}.\n"
                    f"Available: {sorted(keys_in_snap)}"
                )
            cols.append(arr)
            field_names.append(name)
            if name == "u":
                _u_arr = arr

        # Mandatory vector
        hdf_key, comps, _r = _MANDATORY_VECTOR
        arr = _load(hdf_key)
        if arr is None:
            arr = np.zeros((len(positions), 3), dtype=np.float64)
        for i, comp in enumerate(comps):
            cols.append(arr[:, i])
            field_names.append(comp)

        # Optional scalars
        for hdf_key, name, _r in _OPTIONAL_SCALAR:
            arr = _load(hdf_key)
            if arr is not None and arr.ndim == 1:
                cols.append(arr)
                field_names.append(name)
                if name == "ne":
                    _ne_arr = arr

        # Optional vectors
        for hdf_key, comps, _r in _OPTIONAL_VECTOR:
            arr = _load(hdf_key)
            if arr is not None and arr.ndim == 2:
                for i, comp in enumerate(comps):
                    cols.append(arr[:, i])
                    field_names.append(comp)

        # Optional multi-column (expand to indexed scalars)
        for hdf_key, prefix, _r in _OPTIONAL_MULTICOL:
            arr = _load(hdf_key)
            if arr is not None and arr.ndim == 2:
                for i in range(arr.shape[1]):
                    cols.append(arr[:, i])
                    field_names.append(f"{prefix}_{i}")

    # Temperature from InternalEnergy (Kelvin)
    # T = (gamma-1) * u_cgs * mu * m_proton / k_B
    # u is in (unit_vel_cgs)^2 per unit mass; mu from electron abundance if available.
    if _u_arr is not None:
        _GAMMA   = 5.0 / 3.0
        _MP_CGS  = 1.6726e-24   # proton mass [g]
        _KB_CGS  = 1.3806e-16   # Boltzmann [erg/K]
        _XH      = 0.76          # H mass fraction
        u_cgs = _u_arr * unit_vel_cgs ** 2
        if _ne_arr is not None:
            mu = 4.0 / (1.0 + 3.0 * _XH + 4.0 * _XH * _ne_arr)
        else:
            mu = 0.6  # fully ionised solar composition
        temperature = (_GAMMA - 1.0) * u_cgs * mu * _MP_CGS / _KB_CGS
        cols.append(temperature.astype(np.float64, copy=False))
        field_names.append("temperature")

    attrs = np.column_stack(cols).astype(np.float64, copy=False)
    positions = positions.astype(np.float64, copy=False)
    hsml = np.clip(hsml, 0.0, None).astype(np.float64, copy=False)

    scale_label = f"sub-box side={L:.4g}" if box is not None else f"BoxSize={L:.4g}"
    print(
        f"Loaded {len(positions):,} Arepo PartType{part_type} particles"
        f"  ({scale_label} code units, positions renormalised to [0,1]^3)"
    )
    print(
        f"  HSML: {hsml_source}"
        f"  normalised min={hsml.min():.3e}  median={np.median(hsml):.3e}"
        f"  max={hsml.max():.3e}"
    )
    print(f"  Fields ({len(field_names)}): {', '.join(field_names)}")
    return positions, hsml, attrs, field_names, particle_ids


# ─────────────────────────────────────────────────────────────────────────────
# Octree building
# ─────────────────────────────────────────────────────────────────────────────

def build_arepo_octree(
    positions: np.ndarray,
    hsml: np.ndarray,
    attrs: np.ndarray,
    field_names: list[str],
    particle_ids: np.ndarray | None = None,
    *,
    max_depth: int = 12,
    min_depth: int = 2,
    max_members: int = 8,
    max_nodes: int = 200_000_000,
    hsml_geom: str = "sphere",
    split_mass: bool = False,
):
    """
    Build an Octave Octree3D from Arepo gas particles with HSML deposition.

    Parameters
    ----------
    positions    : (N, 3) normalised to [0, 1]^3
    hsml         : (N,)   effective radii in same units
    attrs        : (N, K) per-particle field values
    field_names  : column names for attrs (length K)
    particle_ids : (N,) optional particle IDs passed through to the tree
    max_depth    : finest allowed refinement level  (cell size = 1/2^max_depth)
    min_depth    : coarsest pre-built level
    max_members  : max particles per leaf before splitting
    hsml_geom    : 'sphere' or 'cube' influence region
    split_mass   : divide mass among K intersecting leaves (mass-conserving)

    Returns
    -------
    tree : Octree3D
    """
    oc = _import_octave()

    tree = oc.Octree3D(
        max_depth   = max_depth,
        min_depth   = min_depth,
        max_nodes   = max_nodes,
        max_members = max_members,
    )

    # Register reducers — mass-weighted mean for all intensive fields
    mass_col = field_names.index("mass") if "mass" in field_names else -1
    for i, name in enumerate(field_names):
        # Look up by exact name first, then by prefix (for indexed fields)
        entry = _REDUCERS.get(name)
        if entry is None:
            # indexed fields like passive_scalar_0 → use wmean/mass
            entry = ("wmean", "mass")
        reducer, weight_name = entry
        weight_col = mass_col if (weight_name == "mass" and mass_col >= 0) else None
        if weight_col is not None:
            tree.register_field(name, i, reducer, weight_col)
        else:
            tree.register_field(name, i, reducer)

    # Deposit with HSML
    tree.enable_hsml_deposit(geom=hsml_geom, split_mass=split_mass)
    print(f"Depositing {len(positions):,} particles (hsml_geom={hsml_geom}) …")
    tree.add_points_hsml(positions, hsml, attrs, particle_ids)

    print(
        f"Built octree: {tree.num_leaves():,} leaves  {tree.num_nodes():,} nodes"
        f"  (max_depth={max_depth})"
    )
    return tree


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_arepo_octree_pipeline(
    snap_path: str,
    *,
    part_type: int = 0,
    box=None,
    box_units: str = "code",
    hsml_factor: float = 1.0,
    max_depth: int = 12,
    min_depth: int = 2,
    max_members: int = 8,
    max_nodes: int = 200_000_000,
    hsml_geom: str = "sphere",
    split_mass: bool = False,
    save_path: str | None = None,
):
    """Load an Arepo snapshot, build the HSML-deposited octree, optionally save."""
    positions, hsml, attrs, field_names, pids = load_arepo(
        snap_path, part_type=part_type, box=box, box_units=box_units,
        hsml_factor=hsml_factor,
    )

    print("Building octree …")
    tree = build_arepo_octree(
        positions, hsml, attrs, field_names, pids,
        max_depth   = max_depth,
        min_depth   = min_depth,
        max_members = max_members,
        max_nodes   = max_nodes,
        hsml_geom   = hsml_geom,
        split_mass  = split_mass,
    )

    if save_path is not None:
        tree.write_all_hdf5(save_path, field_names)
        print(f"Octree saved to {save_path}")

    return tree, field_names


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build an Octave octree from an Arepo snapshot using HSML deposition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("snap_path",
        help="Arepo HDF5 snapshot (e.g. snapshot_082.hdf5)")
    parser.add_argument("--save-octree", default=None, metavar="PATH",
        help="Save built octree to this HDF5 file (default: <snap>_octree.h5)")
    parser.add_argument("--part-type", type=int, default=0)

    tg = parser.add_argument_group("octree")
    tg.add_argument("--max-depth",   type=int,   default=12)
    tg.add_argument("--min-depth",   type=int,   default=2)
    tg.add_argument("--max-members", type=int,   default=8,
        help="Max particles per leaf before splitting (8 recommended for moving-mesh)")
    tg.add_argument("--max-nodes",   type=int,   default=200_000_000)

    hg = parser.add_argument_group("HSML")
    hg.add_argument("--hsml-geom",   default="sphere", choices=["sphere","cube"])
    hg.add_argument("--hsml-factor", type=float, default=1.0,
        help="Scale factor on computed/stored HSML (increase to widen influence regions)")
    hg.add_argument("--split-mass",  action="store_true",
        help="Divide mass evenly across K intersecting leaves (mass-conserving)")

    bg = parser.add_argument_group(
        "sub-box",
        "Select a spatial sub-region. Two modes (mutually exclusive):\n"
        "  1) --box XMIN XMAX YMIN YMAX ZMIN ZMAX  — explicit corners in code units\n"
        "  2) --center CX CY CZ --box-side S        — cube centred at (CX,CY,CZ) with side S\n"
        "     All values are in Arepo code units.")
    bg.add_argument("--box", nargs=6, type=float,
        metavar=("XMIN","XMAX","YMIN","YMAX","ZMIN","ZMAX"), default=None,
        help="Explicit sub-box corners in code units")
    bg.add_argument("--center", nargs=3, type=float,
        metavar=("CX","CY","CZ"), default=None,
        help="Centre of the sub-box in code units")
    bg.add_argument("--box-side", type=float, default=None,
        help="Side length of the centred sub-box in code units")

    args = parser.parse_args()

    if args.box is not None and (args.center is not None or args.box_side is not None):
        parser.error("--box and --center/--box-side are mutually exclusive")

    save_path = args.save_octree
    if save_path is None:
        base = os.path.splitext(os.path.basename(args.snap_path))[0]
        save_path = f"{base}_octree.h5"
        print(f"--save-octree not set; saving to {save_path}")

    box, box_units = resolve_box(
        box      = args.box,
        center   = args.center,
        box_side = args.box_side,
    )
    if box is not None:
        print(f"  Sub-box (code units): "
              f"x=[{box[0]:.4g},{box[1]:.4g}]  "
              f"y=[{box[2]:.4g},{box[3]:.4g}]  "
              f"z=[{box[4]:.4g},{box[5]:.4g}]")

    run_arepo_octree_pipeline(
        args.snap_path,
        part_type   = args.part_type,
        box         = box,
        box_units   = box_units,
        hsml_factor = args.hsml_factor,
        max_depth   = args.max_depth,
        min_depth   = args.min_depth,
        max_members = args.max_members,
        max_nodes   = args.max_nodes,
        hsml_geom   = args.hsml_geom,
        split_mass  = args.split_mass,
        save_path   = save_path,
    )


if __name__ == "__main__":
    main()
