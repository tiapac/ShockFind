"""
octree_shockfind_pipeline.py — Python entry point for the Octave-backed ShockFind pipeline.

All heavy computation (octree build, candidate finding, shock characterisation) happens
in C++ via shockfindCore_octave.  This module handles only the preparatory I/O steps:

  1. load_ramses_for_shockfind()  — RAMSES → numpy arrays via yt
  2. run_octree_pipeline()        — orchestrate build + find + characterise
"""

from __future__ import annotations

import os
import sys
import importlib
import subprocess
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports (yt may not be installed in all environments)
# ─────────────────────────────────────────────────────────────────────────────

def _require(name: str, hint: str):
    import importlib.util
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(f"Missing dependency '{name}'. {hint}")
    return importlib.import_module(name)


def _import_shockfind_octave():
    """Import (and optionally build) the shockfindCore_octave extension."""
    try:
        import shockfindCore_octave
        return shockfindCore_octave
    except ImportError:
        pass
    # Try to find the .so in the src directory
    src_dir = os.path.dirname(__file__)
    so_dir  = os.path.join(src_dir, "shockfindCore_octave")
    build_dir = os.path.join(so_dir, "build")
    if not os.path.isdir(so_dir):
        raise RuntimeError(
            "shockfindCore_octave not found. Build it with:\n"
            f"  cmake -B {build_dir} -S {so_dir} && cmake --build {build_dir} -j$(nproc)\n"
            "Then install or add the build dir to PYTHONPATH."
        )
    raise ImportError(
        "shockfindCore_octave extension not importable. Build it first:\n"
        f"  cmake -B {build_dir} -S {so_dir} "
        f"-DOCTAVE_SRC_DIR=/path/to/Octave/src\n"
        f"  cmake --build {build_dir} -j$(nproc)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Required MHD field names (order defines the attribute column layout)
# ─────────────────────────────────────────────────────────────────────────────

SHOCKFIND_FIELDS = ["density", "pressure", "vx", "vy", "vz", "bx", "by", "bz"]

# Mapping from ShockFind canonical names to common RAMSES yt field names
_RAMSES_FIELD_MAP = {
    "density":  [("gas","density")],
    "pressure": [("gas","pressure"), ("gas","Pressure")],
    "vx":       [("gas","velocity_x"), ("gas","x-velocity")],
    "vy":       [("gas","velocity_y"), ("gas","y-velocity")],
    "vz":       [("gas","velocity_z"), ("gas","z-velocity")],
    "bx":       [("gas","magnetic_field_x"), ("gas","Bx"), ("gas","bfield_x")],
    "by":       [("gas","magnetic_field_y"), ("gas","By"), ("gas","bfield_y")],
    "bz":       [("gas","magnetic_field_z"), ("gas","Bz"), ("gas","bfield_z")],
}


def load_ramses_for_shockfind(
    info_path: str,
    box=None,
    box_units: str = "code",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a RAMSES snapshot and extract gas cell centres + MHD field values.

    Parameters
    ----------
    info_path : str
        Path to RAMSES output directory or info_*.txt file.
    box : sequence of 6 floats or None
        Optional bounding box [xmin, xmax, ymin, ymax, zmin, zmax].
    box_units : 'code' | 'normalized'
        Units for the box coordinates.

    Returns
    -------
    positions : (N, 3) float64 — cell centres normalised to [0,1]^3
    attrs     : (N, 8) float64 — columns in SHOCKFIND_FIELDS order
    """
    yt = _require("yt", "Install yt: pip install yt (or conda install -c conda-forge yt)")

    # Find info file
    if os.path.isfile(info_path):
        info = info_path
    else:
        import glob
        candidates = sorted(glob.glob(os.path.join(info_path, "info_*.txt")))
        if not candidates:
            raise FileNotFoundError(f"No info_*.txt under {info_path}")
        info = candidates[-1]

    ds = yt.load(info)
    ad = ds.all_data()

    le = ds.domain_left_edge.to("code_length").v
    w  = ds.domain_width   .to("code_length").v

    x = ad[("gas","x")].to("code_length").v
    y = ad[("gas","y")].to("code_length").v
    z = ad[("gas","z")].to("code_length").v

    positions = np.column_stack([
        np.clip((x - le[0]) / w[0], 0.0, 1.0),
        np.clip((y - le[1]) / w[1], 0.0, 1.0),
        np.clip((z - le[2]) / w[2], 0.0, 1.0),
    ])

    # Optional box filter
    if box is not None:
        bx = np.array(box, dtype=float)
        if box_units == "code":
            bx[:2] = (bx[:2] - le[0]) / w[0]
            bx[2:4] = (bx[2:4] - le[1]) / w[1]
            bx[4:] = (bx[4:] - le[2]) / w[2]
        mask = (
            (positions[:, 0] >= bx[0]) & (positions[:, 0] <= bx[1]) &
            (positions[:, 1] >= bx[2]) & (positions[:, 1] <= bx[3]) &
            (positions[:, 2] >= bx[4]) & (positions[:, 2] <= bx[5])
        )
        positions = positions[mask]

    # Gather field columns
    cols = []
    for name in SHOCKFIND_FIELDS:
        val = None
        for yt_key in _RAMSES_FIELD_MAP[name]:
            if yt_key in ds.field_list:
                arr = np.asarray(ad[yt_key], dtype=float)
                if box is not None:
                    arr = arr[mask]
                val = arr
                break
        if val is None:
            print(f"Warning: field '{name}' not found in RAMSES dataset — filling with zeros")
            val = np.zeros(positions.shape[0], dtype=float)
        cols.append(val)

    attrs = np.column_stack(cols)
    print(f"Loaded {positions.shape[0]:,} RAMSES cells with {len(SHOCKFIND_FIELDS)} MHD fields")
    return positions.astype(np.float64, copy=False), attrs.astype(np.float64, copy=False)


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_octree_pipeline(
    info_path: str,
    *,
    box=None,
    box_units: str = "code",
    max_depth: int = 14,
    min_depth: int = 3,
    max_members: int = 1,
    max_nodes: int = 100_000_000,
    div_threshold: float = -0.1,
    grad_threshold: float = 0.1,
    shock_params: dict | None = None,
    quiet: bool = False,
):
    """
    End-to-end pipeline: RAMSES → octree → candidates → shock results.

    Parameters
    ----------
    info_path     : RAMSES output directory or info file
    box / box_units : optional spatial sub-region
    max_depth, min_depth, max_members, max_nodes : octree parameters
      max_members=1 preserves the RAMSES AMR structure exactly.
    div_threshold : candidate threshold on div(v) (negative = converging flow)
    grad_threshold: candidate threshold on |∇ρ|/ρ (dimensionless)
    shock_params  : dict forwarded to characterise_shocks_octree (same as `extra` dict)
    quiet         : suppress per-candidate progress output

    Returns
    -------
    handle   : OctreeHandle — the built octree (reusable for further queries)
    candidates: list[int]  — leaf node indices that passed the candidate filter
    results  : (data, header) tuple — 17-array shock result in standard ShockFind layout
    """
    sco = _import_shockfind_octave()

    # 1. Load RAMSES
    positions, attrs = load_ramses_for_shockfind(info_path, box=box, box_units=box_units)

    # 2. Build octree
    print("Building octree...")
    handle = sco.build_octree(
        positions, attrs, SHOCKFIND_FIELDS,
        max_depth=max_depth,
        min_depth=min_depth,
        max_members=max_members,
        max_nodes=max_nodes,
    )
    print(f"Octree: {handle.num_leaves:,} leaves, {handle.num_nodes:,} nodes")

    # 3. Find candidates
    print("Finding shock candidates...")
    candidates = sco.find_candidates(handle, div_threshold, grad_threshold)
    print(f"Found {len(candidates):,} candidate leaves")

    # 4. Characterise shocks
    print("Characterising shocks...")
    params = shock_params or {}
    results = sco.characterise_shocks_octree(handle, candidates, params, quiet)

    return handle, candidates, results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point (for quick testing)
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Octave-backed ShockFind pipeline")
    parser.add_argument("data_path", help="RAMSES output directory or info_*.txt")
    parser.add_argument("--max-depth",   type=int,   default=14)
    parser.add_argument("--min-depth",   type=int,   default=3)
    parser.add_argument("--max-members", type=int,   default=1)
    parser.add_argument("--max-nodes",   type=int,   default=100_000_000)
    parser.add_argument("--div-threshold",  type=float, default=-0.1,
                        help="Divergence threshold for candidates (default: -0.1)")
    parser.add_argument("--grad-threshold", type=float, default=0.1,
                        help="Dimensionless density gradient threshold (default: 0.1)")
    parser.add_argument("--quiet",  action="store_true")
    args = parser.parse_args()

    handle, candidates, results = run_octree_pipeline(
        args.data_path,
        max_depth=args.max_depth,
        min_depth=args.min_depth,
        max_members=args.max_members,
        max_nodes=args.max_nodes,
        div_threshold=args.div_threshold,
        grad_threshold=args.grad_threshold,
        quiet=args.quiet,
    )
    data, header = results
    print(f"Characterised {len(data[0])} shocks")
    # Print a simple summary of non-flagged results
    flags = np.asarray(data[16])
    families = np.asarray(data[6])
    print(f"  OK (flag=0): {(flags==0).sum()}")
    print(f"  Fast (family=12): {(families==12).sum()}")
    print(f"  Slow (family=34): {(families==34).sum()}")


if __name__ == "__main__":
    main()
