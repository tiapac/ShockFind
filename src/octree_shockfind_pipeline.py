"""
octree_shockfind_pipeline.py — CLI and library entry point for the Octave-backed ShockFind pipeline.

Mirrors the workflow of main_example.py but uses Octave's AMR octree instead of a
uniform covering grid.  Heavy computation (tree build, candidates, characterisation)
is fully in C++ via shockfindCore_octave.  Results are saved/loaded/plotted through
the existing shock_finder machinery so all downstream analysis tools work unchanged.

Usage (standalone):
    python src/octree_shockfind_pipeline.py /path/to/ramses/output_00021 \\
        --output /path/to/results/ --name my_run

Usage (library):
    from src.octree_shockfind_pipeline import run_octree_pipeline
    handle, sf = run_octree_pipeline("/path/to/output_00021")
    sf.save_results(path="results/", name="run01")
    sf.plot3D()
"""

from __future__ import annotations

import os
import sys
import importlib
import numpy as np
import pickle

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

def _require(name: str, hint: str):
    import importlib.util
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(f"Missing dependency '{name}'. {hint}")
    return importlib.import_module(name)


def _import_shockfind_octave():
    """Import the shockfindCore_octave C++ extension."""
    # Try installed / on PYTHONPATH first
    try:
        import shockfindCore_octave
        return shockfindCore_octave
    except ImportError:
        pass
    # Try the build directory next to this file
    src_dir = os.path.dirname(os.path.abspath(__file__))
    build_so = os.path.join(src_dir, "shockfindCore_octave", "build")
    if os.path.isdir(build_so):
        sys.path.insert(0, build_so)
        try:
            import shockfindCore_octave
            return shockfindCore_octave
        except ImportError:
            sys.path.pop(0)
    raise ImportError(
        "shockfindCore_octave not found. Build it with:\n"
        "  ./setup.sh --octave [--octave-src /path/to/Octave/src]"
    )


def _import_shock_finder():
    """Import shock_finder, trying both installed and local paths."""
    try:
        from ShockFind import shock_finder as sf_cls
        return sf_cls
    except ImportError:
        pass
    try:
        from src.shockfind_interface import shock_finder as sf_cls
        return sf_cls
    except ImportError:
        pass
    # Running from ShockFind root
    src_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(src_dir))
    from src.shockfind_interface import shock_finder as sf_cls
    return sf_cls


# ─────────────────────────────────────────────────────────────────────────────
# MHD field layout (order = attribute column index in the attrs matrix)
# ─────────────────────────────────────────────────────────────────────────────

SHOCKFIND_FIELDS = ["density", "pressure", "vx", "vy", "vz", "bx", "by", "bz"]

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


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_ramses_for_shockfind(
    info_path: str,
    box=None,
    box_units: str = "code",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load RAMSES gas cell centres + MHD field values via yt.

    Returns
    -------
    positions : (N, 3) float64 — cell centres normalised to [0,1]^3
    attrs     : (N, 8) float64 — columns in SHOCKFIND_FIELDS order
    """
    yt = _require("yt", "Install yt: pip install yt (or conda install -c conda-forge yt)")

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

    mask = None
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

    cols = []
    for name in SHOCKFIND_FIELDS:
        val = None
        for yt_key in _RAMSES_FIELD_MAP[name]:
            if yt_key in ds.field_list:
                arr = np.asarray(ad[yt_key], dtype=float)
                if mask is not None:
                    arr = arr[mask]
                val = arr
                break
        if val is None:
            print(f"Warning: field '{name}' not found in RAMSES dataset — filling with zeros")
            val = np.zeros(positions.shape[0], dtype=float)
        cols.append(val)

    attrs = np.column_stack(cols)
    print(f"Loaded {positions.shape[0]:,} RAMSES cells  ({len(SHOCKFIND_FIELDS)} MHD fields)")
    return positions.astype(np.float64, copy=False), attrs.astype(np.float64, copy=False)


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline — returns (OctreeHandle, shock_finder)
# ─────────────────────────────────────────────────────────────────────────────

def run_octree_pipeline(
    info_path: str,
    name: str = "octree_run",
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

    Returns
    -------
    handle : OctreeHandle  — built octree (reusable for further queries)
    sf     : shock_finder  — populated with results; call sf.save_results(),
                             sf.plot3D(), sf.histograms() etc. as in main_example.py
    """
    sco = _import_shockfind_octave()
    shock_finder = _import_shock_finder()

    # 1. Load RAMSES data
    positions, attrs = load_ramses_for_shockfind(info_path, box=box, box_units=box_units)

    # 2. Build octree
    print("Building octree …")
    handle = sco.build_octree(
        positions, attrs, SHOCKFIND_FIELDS,
        max_depth=max_depth,
        min_depth=min_depth,
        max_members=max_members,
        max_nodes=max_nodes,
    )
    print(f"  {handle.num_leaves:,} leaves  {handle.num_nodes:,} nodes  "
          f"(dx = 1/{1 << handle.max_depth} = {handle.cell_size:.3e})")

    # 3. Find shock candidates
    print("Finding candidates …")
    candidates = sco.find_candidates(handle, div_threshold, grad_threshold)
    print(f"  {len(candidates):,} candidate leaves")

    # 4. Characterise shocks
    print("Characterising shocks …")
    params = shock_params or {}
    data, header = sco.characterise_shocks_octree(handle, candidates, params, quiet)

    # 5. Wire into shock_finder for save/load/plot compatibility
    sf = shock_finder(name=name)
    sf.shocks = data
    sf.header = header
    # dx in physical units: positions are in [0,1]^3 on a 2^max_depth grid
    sf.shocks_data(dx=handle.cell_size)

    return handle, sf


# ─────────────────────────────────────────────────────────────────────────────
# CLI — mirrors main_example.py usage
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="ShockFind on a RAMSES output using Octave AMR octree (no uniform grid)")
    parser.add_argument("data_path",
        help="RAMSES output directory (e.g. output_00021/) or info_*.txt path")
    parser.add_argument("-out", "--output", default=None,
        help="Directory to save results (default: <data_path>/shockfind_results/)")
    parser.add_argument("-name", "--name", default=None,
        help="Run name used for the output file (default: derived from data_path)")
    parser.add_argument("--max-depth",    type=int,   default=14)
    parser.add_argument("--min-depth",    type=int,   default=3)
    parser.add_argument("--max-members",  type=int,   default=1,
        help="Max points per octree leaf; 1 mirrors RAMSES AMR structure exactly")
    parser.add_argument("--max-nodes",    type=int,   default=100_000_000)
    parser.add_argument("--div-threshold",  type=float, default=-0.1,
        help="div(v) threshold for candidates (default: -0.1, dimensionless)")
    parser.add_argument("--grad-threshold", type=float, default=0.1,
        help="|∇ρ|/ρ threshold for candidates (default: 0.1, dimensionless)")
    parser.add_argument("--gamma", type=float, default=5.0/3.0)
    parser.add_argument("--no-periodic", action="store_true",
        help="Treat domain boundaries as non-periodic")
    parser.add_argument("-load", "--load", action="store_true",
        help="Load previously saved results instead of running the analysis")
    parser.add_argument("--plot",  action="store_true", help="Show 3D shock plot after analysis")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Resolve output dir and run name
    data_path = os.path.abspath(args.data_path)
    if args.output:
        results_path = args.output
    else:
        results_path = os.path.join(os.path.dirname(data_path), "shockfind_results")

    if args.name:
        run_name = args.name
    else:
        run_name = "octree_" + os.path.basename(data_path.rstrip("/"))

    os.makedirs(results_path, exist_ok=True)

    shock_finder = _import_shock_finder()

    if args.load:
        sf = shock_finder(name=run_name)
        sf.load_results(path=results_path, name=run_name)
        print(f"Loaded results from {results_path}/{run_name}_result.pk")
        handle = None
    else:
        shock_params = {
            "gamma":    args.gamma,
            "periodic": [not args.no_periodic] * 3,
        }
        handle, sf = run_octree_pipeline(
            data_path,
            name=run_name,
            max_depth=args.max_depth,
            min_depth=args.min_depth,
            max_members=args.max_members,
            max_nodes=args.max_nodes,
            div_threshold=args.div_threshold,
            grad_threshold=args.grad_threshold,
            shock_params=shock_params,
            quiet=args.quiet,
        )
        sf.save_results(path=results_path, name=run_name)
        print(f"Results saved to {results_path}/{run_name}_result.pk")

    # Summary
    shocks = sf.shocks
    flags    = np.asarray(shocks[16])
    families = np.asarray(shocks[6])
    total    = len(flags)
    ok       = int((flags == 0).sum())
    fast     = int((families == 12).sum())
    slow     = int((families == 34).sum())
    print(f"\nTotal characterised: {total}  |  OK (flag=0): {ok}  |  Fast: {fast}  |  Slow: {slow}")

    if args.plot:
        import matplotlib.pyplot as plt
        ax, fig = sf.plot3D(types="fs")
        fig_path = os.path.join(results_path, f"{run_name}_3Dplot.png")
        fig.savefig(fig_path, dpi=150)
        print(f"3D plot saved to {fig_path}")
        plt.show()


if __name__ == "__main__":
    main()
