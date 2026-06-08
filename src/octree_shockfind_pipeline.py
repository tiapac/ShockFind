"""
octree_shockfind_pipeline.py — CLI and library entry point for the Octave-backed ShockFind pipeline.

Mirrors the workflow of main_example.py but uses Octave's AMR octree instead of a
uniform covering grid.  Heavy computation (tree build, candidates, characterisation)
is fully in C++ via shockfindCore_octave.  Results are saved/loaded/plotted through
the existing shock_finder machinery so all downstream analysis tools work unchanged.

Data loading is delegated to src/interfaces/ — pass --interface to select the code.
SPH / moving-mesh codes (Arepo) require a pre-built octree HDF5 via --load-octree;
run src/build_octree.py first to produce it.

Usage (standalone):
    # RAMSES — build octree and run shockfind in one shot:
    python src/octree_shockfind_pipeline.py output_00021/ \\
        --interface ramses --output results/ --name run21

    # Arepo — build octree first, then run shockfind:
    python src/build_octree.py arepo snapshot_082.hdf5 --save-octree oct.h5
    python src/octree_shockfind_pipeline.py --load-octree oct.h5

Usage (library):
    from src.octree_shockfind_pipeline import run_octree_pipeline
    handle, sf = run_octree_pipeline("output_00021", interface="ramses")
    sf.save_results(path="results/", name="run01")
    sf.plot3D()
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pickle

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

def _require(name: str, hint: str):
    import importlib.util
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(f"Missing dependency '{name}'. {hint}")
    return __import__(name)


def _import_shockfind_octave():
    """Import the shockfindCore_octave C++ extension."""
    try:
        import shockfindCore_octave
        return shockfindCore_octave
    except ImportError:
        pass
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
    """Import shock_finder from the ShockFind package."""
    src_dir        = os.path.dirname(os.path.abspath(__file__))
    shockfind_root = os.path.dirname(src_dir)
    pkg_root       = os.path.dirname(shockfind_root)
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from ShockFind import shock_finder as sf_cls
    return sf_cls


# ─────────────────────────────────────────────────────────────────────────────
# Field layouts — imported from interfaces (re-exported for backward compat)
# ─────────────────────────────────────────────────────────────────────────────

from src.interfaces import (   # noqa: E402
    SHOCKFIND_FIELDS_MHD,
    SHOCKFIND_FIELDS_HYDRO,
    SHOCKFIND_FIELDS,
    get_interface,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — thin wrappers over src/interfaces/
# ─────────────────────────────────────────────────────────────────────────────

def load_ramses_for_shockfind(
    info_path: str,
    box=None,
    box_units: str = "code",
    hydro_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load RAMSES cells via yt.  Backward-compatible wrapper over RamsesInterface.

    Returns (positions, attrs, field_names).
    """
    iface = get_interface(
        "ramses", info_path,
        box=box, box_units=box_units, hydro_only=hydro_only,
    )
    data = iface.load()
    return data.positions, data.attrs, data.field_names


def _load_for_pipeline(
    interface: str,
    data_path: str,
    box=None,
    box_units: str = "code",
    hydro_only: bool = False,
    **iface_kwargs,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load any interface and return (positions, attrs, field_names).

    Raises clearly for HSML codes that cannot go straight to shockfindCore_octave.
    """
    iface = get_interface(
        interface, data_path,
        box=box, box_units=box_units, hydro_only=hydro_only,
        **iface_kwargs,
    )
    if iface.uses_hsml:
        raise ValueError(
            f"Interface '{interface}' uses HSML deposition, which requires a "
            "pre-built octree HDF5.\n"
            "  Step 1: python src/build_octree.py {interface} {data_path} "
            "--save-octree my_oct.h5\n"
            "  Step 2: python src/octree_shockfind_pipeline.py "
            "--load-octree my_oct.h5"
        )
    data = iface.load()
    return data.positions, data.attrs, data.field_names


# ─────────────────────────────────────────────────────────────────────────────
# Candidate finding — mirrors shock_finder.set_thresholds + find_candidates
# ─────────────────────────────────────────────────────────────────────────────

def _find_candidates_full(
    sco,
    handle,
    *,
    # ── Physical threshold mode (mirrors shock_finder.set_thresholds) ─────────
    # vshock_min and rhomean must be in the same units as the loaded velocity/density.
    # dx defaults to handle.cell_size (normalised finest-level spacing).
    vshock_min: float | None = None,
    rhomean: float | None = None,
    dx: float | None = None,
    N_cells: int = 3,
    compress: float = 1.1,
    # ── Raw/normalised threshold mode (kept for future unit-normalised use) ───
    # div_threshold : div(v) < div_threshold  (negative, e.g. -0.1)
    # grad_threshold: |∇ρ|/ρ > grad_threshold (dimensionless, e.g. 0.1)
    div_threshold: float | None = None,
    grad_threshold: float | None = None,
    # ── Additional filters (mirrors shock_finder.find_candidates) ─────────────
    mach_min: float | None = 1.5,
    vmag_min: float | None = None,
    use_gradTRho: bool = True,
    gamma: float = 5. / 3.,
    hydro_only: bool = False,
) -> list[int]:
    """
    Find shock candidate leaves — full mirror of the regular-grid shock_finder workflow.

    Physical mode  (vshock_min / rhomean given):
      Computes convergence and density-gradient thresholds from physical quantities,
      exactly as shock_finder.set_thresholds() does for the uniform grid.
      dx should be in the same length units as the spatial derivatives (normalised
      [0,1]^3 by default, i.e. handle.cell_size).

    Raw/normalised mode  (div_threshold / grad_threshold given, no vshock_min):
      Uses the dimensionless thresholds directly — suitable for unit-normalised
      fields and reserved for future normalisation of the main ShockFind library.

    Additional filters applied in both modes when requested:
      mach_min    : Mach > mach_min  (same formula as shock_finder.find_candidates)
      vmag_min    : |v| > vmag_min
      use_gradTRho: ∇T·∇ρ > 0  (T ∝ P/ρ; sign is unit-independent)

    Returns
    -------
    list[int] — node indices of candidate leaves (ready for characterise_shocks_octree).
    """
    rho   = np.asarray(handle.rho_arr)
    pres  = np.asarray(handle.pres_arr)
    vx    = np.asarray(handle.vx_arr)
    vy    = np.asarray(handle.vy_arr)
    vz    = np.asarray(handle.vz_arr)
    div_v = np.asarray(handle.div_v_arr)
    gx    = np.asarray(handle.grad_rho_x)
    gy    = np.asarray(handle.grad_rho_y)
    gz    = np.asarray(handle.grad_rho_z)
    # B arrays are empty in hydro runs (handle.has_bfield == False)
    if not hydro_only:
        bx = np.asarray(handle.bx_arr)
        by = np.asarray(handle.by_arr)
        bz = np.asarray(handle.bz_arr)

    rho_safe = np.where(np.abs(rho) > 1e-30, np.abs(rho), 1e-30)

    if dx is None:
        dx = handle.cell_size

    # ── Convergence filter ────────────────────────────────────────────────────
    if vshock_min is not None:
        # Physical: convergenge_threshold = vshock_min * (compress-1) / (compress * N * dx)
        # div_v < 0 is converging; candidates must exceed this magnitude.
        conv_thr = vshock_min * (compress - 1.0) / (compress * N_cells * dx)
        div_filter = div_v < -conv_thr
    elif div_threshold is not None:
        div_filter = div_v < div_threshold          # div_threshold is negative (e.g. -0.1)
    else:
        raise ValueError("Provide either vshock_min or div_threshold")

    candidates = div_filter

    # ── Density gradient filter ───────────────────────────────────────────────
    grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    if rhomean is not None:
        # Physical: nablaRho_threshold = rhomean * (compress-1) / (N * dx)
        nabla_rho_thr = rhomean * (compress - 1.0) / (N_cells * dx)
        candidates = candidates & (grad_mag > nabla_rho_thr)
    elif grad_threshold is not None:
        # Normalised: |∇ρ|/ρ > grad_threshold
        candidates = candidates & ((grad_mag / rho_safe) > grad_threshold)
    # (if neither rhomean nor grad_threshold: no gradient filter applied)

    # ── ∇T·∇ρ dot product filter (replaces grad filter when use_gradTRho=True) ─
    if use_gradTRho:
        T_prop = pres / rho_safe                    # proportional to T; sign of gradient is unit-independent
        gTx, gTy, gTz = (np.asarray(a) for a in sco.compute_gradient_of(handle, T_prop))
        grad_T_dot_rho = gTx * gx + gTy * gy + gTz * gz
        candidates = candidates & (grad_T_dot_rho > 0.0)

    # ── Mach number filter ────────────────────────────────────────────────────
    if mach_min is not None and mach_min > 0.0:
        cs   = np.sqrt(gamma * np.maximum(pres, 0.0) / rho_safe)
        vmag = np.sqrt(vx**2 + vy**2 + vz**2)
        if hydro_only:
            # Sonic Mach only — no Alfvén speed
            mach = vmag / np.maximum(cs, 1e-30)
        else:
            bmag = np.sqrt(bx**2 + by**2 + bz**2)
            ca   = bmag / np.sqrt(4.0 * np.pi * rho_safe)
            mach = vmag / np.maximum(0.7 * np.minimum(cs, ca), 1e-30)
        candidates = candidates & (mach >= mach_min)

    # ── Velocity magnitude filter ─────────────────────────────────────────────
    if vmag_min is not None and vmag_min > 0.0:
        vmag = np.sqrt(vx**2 + vy**2 + vz**2)
        candidates = candidates & (vmag >= vmag_min)

    node_indices = np.asarray(handle.leaf_node_indices)
    return [int(nid) for nid in node_indices[candidates]]


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline — returns (OctreeHandle, shock_finder)
# ─────────────────────────────────────────────────────────────────────────────

def run_octree_pipeline(
    info_path: str | None = None,
    name: str = "octree_run",
    *,
    interface: str = "ramses",
    box=None,
    box_units: str = "code",
    # ── Tree parameters (ignored when octree_load_path is given) ─────────────
    max_depth: int = 14,
    min_depth: int = 3,
    max_members: int = 1,
    max_nodes: int = 100_000_000,
    # ── Octree HDF5 persistence ───────────────────────────────────────────────
    octree_save_path: str | None = None,
    octree_load_path: str | None = None,
    # ── Candidate finding — physical mode ─────────────────────────────────────
    vshock_min: float | None = None,
    rhomean: float | None = None,
    dx: float | None = None,
    N_cells: int = 3,
    compress: float = 1.1,
    # ── Candidate finding — raw/normalised mode ───────────────────────────────
    div_threshold: float = -0.1,
    grad_threshold: float = 0.1,
    # ── Additional candidate filters ──────────────────────────────────────────
    mach_min: float | None = 1.5,
    vmag_min: float | None = None,
    use_gradTRho: bool = True,
    # ── Physics mode ──────────────────────────────────────────────────────────
    hydro_only: bool = False,
    # ── Characterisation parameters (mirror of shock_finder.extra_params) ─────
    gamma: float = 5. / 3.,
    periodic: list | None = None,
    method_norm: str = "point_gradient",
    method_plane: str = "point_field",
    Rgrad: int = 3,
    Rcylinder: int = 3,
    line_range: int = 10,
    field_ref: int = 0,
    shock_ratio: float = 1.1,
    # ── Analysis control ──────────────────────────────────────────────────────
    rescale: int = 0,
    quiet: bool = False,
):
    """
    End-to-end pipeline: dataset → octree → candidates → shock results.

    ``interface`` selects the data loader ('ramses', 'arepo', …).
    HSML-based codes (Arepo) require a pre-built octree via ``octree_load_path``;
    run ``src/build_octree.py`` first to produce it.

    Mirrors main_example.py: physical thresholds (vshock_min / rhomean) are used
    when provided; otherwise raw dimensionless thresholds (div_threshold / grad_threshold)
    are used — kept for unit-normalised fields and future library normalisation.

    octree_save_path : if given, the built octree is saved to this HDF5 file after build.
    octree_load_path : if given, the octree is loaded from this HDF5 file and data
                       loading / tree building are skipped entirely (info_path not needed).

    Returns
    -------
    handle : OctreeHandle  — built octree (reusable for further queries)
    sf     : shock_finder  — populated with results; call sf.save_results(),
                             sf.plot3D(), sf.histograms() etc. as in main_example.py
    """
    if octree_load_path is None and info_path is None:
        raise ValueError("Provide either info_path (data path) or octree_load_path (HDF5)")

    sco = _import_shockfind_octave()
    shock_finder = _import_shock_finder()

    if periodic is None:
        periodic = [False, False, False]

    if octree_load_path is not None:
        # Load pre-built octree — skip RAMSES loading and tree construction
        print(f"Loading octree from {octree_load_path} …")
        handle = sco.load_octree(octree_load_path)
        print(f"  {handle.num_leaves:,} leaves  {handle.num_nodes:,} nodes  "
              f"(dx = 1/{1 << handle.max_depth} = {handle.cell_size:.3e})")
    else:
        # 1. Load data via the selected interface
        positions, attrs, field_names = _load_for_pipeline(
            interface, info_path,
            box=box, box_units=box_units, hydro_only=hydro_only,
        )

        # 2. Build octree
        print("Building octree …")
        handle = sco.build_octree(
            positions, attrs, field_names,
            max_depth=max_depth,
            min_depth=min_depth,
            max_members=max_members,
            max_nodes=max_nodes,
        )
        print(f"  {handle.num_leaves:,} leaves  {handle.num_nodes:,} nodes  "
              f"(dx = 1/{1 << handle.max_depth} = {handle.cell_size:.3e})")

        if octree_save_path is not None:
            if os.path.isdir(octree_save_path):
                octree_save_path = os.path.join(octree_save_path, f"{name}_octree.h5")
            sco.save_octree(handle, octree_save_path)
            print(f"  Octree saved to {octree_save_path}")

    # 3. Find shock candidates
    print("Finding candidates …")
    candidates = _find_candidates_full(
        sco, handle,
        vshock_min=vshock_min,
        rhomean=rhomean,
        dx=dx,
        N_cells=N_cells,
        compress=compress,
        div_threshold=div_threshold,
        grad_threshold=grad_threshold,
        mach_min=mach_min,
        vmag_min=vmag_min,
        use_gradTRho=use_gradTRho,
        gamma=gamma,
        hydro_only=hydro_only,
    )
    print(f"  {len(candidates):,} candidate leaves")

    # rescale: subsample candidates (mirrors shock_finder.analyse_candidates rescale)
    for _ in range(rescale):
        candidates = candidates[::2]
    if rescale > 0:
        print(f"  {len(candidates):,} candidates after rescale={rescale}")

    # 4. Characterise shocks
    print("Characterising shocks …")
    shock_params = {
        "gamma":        gamma,
        "periodic":     periodic,
        "method_norm":  method_norm,
        "method_plane": method_plane,
        "Rgrad":        Rgrad,
        "Rcylinder":    Rcylinder,
        "line_range":   line_range,
        "field_ref":    field_ref,
        "shock_ratio":  shock_ratio,
        "hydro_only":   hydro_only,
    }
    data, header = sco.characterise_shocks_octree(handle, candidates, shock_params, quiet)

    # 5. Wire into shock_finder for save/load/plot compatibility
    sf = shock_finder(name=name)
    sf.shocks = data
    sf.header = header
    sf.shocks_data(dx=handle.cell_size)

    return handle, sf


# ─────────────────────────────────────────────────────────────────────────────
# CLI — mirrors main_example.py usage
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="ShockFind using the Octave AMR octree (dataset-agnostic).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_path", nargs="?", default=None,
        help="Dataset path (RAMSES directory, Arepo HDF5, …). "
             "Not required when --load-octree is given.")
    parser.add_argument("--interface", default="ramses",
        choices=["ramses", "arepo"],
        help="Simulation code / data format to load.")
    parser.add_argument("-out", "--output", default=None,
        help="Directory to save results (default: <data_path>/shockfind_results/)")
    parser.add_argument("-name", "--name", default=None,
        help="Run name used for the output file (default: derived from data_path)")

    # ── Tree parameters ───────────────────────────────────────────────────────
    tree = parser.add_argument_group("octree parameters")
    tree.add_argument("--max-depth",    type=int,   default=14)
    tree.add_argument("--min-depth",    type=int,   default=3)
    tree.add_argument("--max-members",  type=int,   default=1,
        help="Max points per octree leaf; 1 mirrors RAMSES AMR structure exactly")
    tree.add_argument("--max-nodes",    type=int,   default=100_000_000)

    # ── Candidate finding — physical mode (mirrors set_thresholds) ───────────
    cand = parser.add_argument_group(
        "candidate thresholds — physical mode",
        "Mirrors shock_finder.set_thresholds(). Provide vshock_min + rhomean to use "
        "physical thresholds; otherwise raw dimensionless thresholds are used.")
    cand.add_argument("--vshock-min", type=float, default=1e5,
        help="Minimum shock velocity in dataset velocity units (e.g. cm/s for CGS). "
             "Determines the convergence threshold.")
    cand.add_argument("--rhomean", type=float, default=None,
        help="Mean pre-shock density in dataset density units. "
             "Determines the density-gradient threshold.")
    cand.add_argument("--dx", type=float, default=None,
        help="Reference cell size for threshold computation (normalised [0,1] units). "
             "Defaults to handle.cell_size = 1/2^max_depth.")
    cand.add_argument("--N-cells", type=int, default=3,
        help="Number of cells for shock spreading in threshold formula.")
    cand.add_argument("--compress", type=float, default=1.1,
        help="Expected density compression ratio for threshold formula.")

    # ── Candidate finding — raw/normalised mode ───────────────────────────────
    raw = parser.add_argument_group(
        "candidate thresholds — raw/normalised mode",
        "Used when --vshock-min / --rhomean are not given. "
        "Suitable for unit-normalised fields.")
    raw.add_argument("--div-threshold",  type=float, default=-0.1,
        help="div(v) threshold; candidates have div(v) < this value (must be negative).")
    raw.add_argument("--grad-threshold", type=float, default=0.1,
        help="|∇ρ|/ρ threshold; candidates must exceed this (dimensionless).")

    # ── Additional candidate filters (mirrors find_candidates) ───────────────
    filt = parser.add_argument_group("additional candidate filters")
    filt.add_argument("--mach-min", type=float, default=1.5,
        help="Minimum Mach number filter; same formula as shock_finder.find_candidates.")
    filt.add_argument("--vmag-min", type=float, default=None,
        help="Minimum velocity magnitude filter (dataset velocity units).")
    filt.add_argument("--no-gradTRho", action="store_true",
        help="Disable ∇T·∇ρ > 0 filter (enabled by default, mirrors use_gradTRho=True).")
    filt.add_argument("--hydro-only", action="store_true",
        help="Pure-hydro mode: skip magnetic field loading and use sonic-only "
             "shock classification (family=1 instead of fast=12/slow=34).")

    # ── Characterisation parameters (mirrors extra_params + analyse_candidates) ─
    char = parser.add_argument_group("characterisation parameters")
    char.add_argument("--gamma", type=float, default=1.666667,
        help="Adiabatic index.")
    char.add_argument("--no-periodic", action="store_true",
        help="Treat domain boundaries as non-periodic.")
    char.add_argument("--method-norm",  default="point_gradient",
        choices=["point_gradient", "average_gradient"],
        help="Method for shock normal estimation.")
    char.add_argument("--method-plane", default="point_field",
        choices=["point_field", "average_field"],
        help="Method for shock plane field.")
    char.add_argument("--Rgrad",       type=int,   default=3)
    char.add_argument("--Rcylinder",   type=int,   default=3)
    char.add_argument("--line-range",  type=int,   default=10)
    char.add_argument("--shock-ratio", type=float, default=1.1)

    # ── Octree HDF5 persistence ───────────────────────────────────────────────
    persist = parser.add_argument_group("octree persistence (HDF5)")
    persist.add_argument("--save-octree", type=str, default=None, metavar="PATH",
        help="Save the built octree to this HDF5 file (requires USE_HDF5 build). "
             "Useful to skip the expensive RAMSES+build step on re-runs.")
    persist.add_argument("--load-octree", type=str, default=None, metavar="PATH",
        help="Load octree from this HDF5 file instead of building from RAMSES. "
             "data_path is not required when this flag is given.")

    # ── Analysis control ──────────────────────────────────────────────────────
    ctrl = parser.add_argument_group("analysis control")
    ctrl.add_argument("--rescale", type=int, default=0,
        help="Subsample candidates: keep every 2^rescale-th candidate (mirrors "
             "shock_finder.analyse_candidates rescale).")
    ctrl.add_argument("-load", "--load", action="store_true",
        help="Load previously saved results instead of running the analysis.")
    ctrl.add_argument("--plot",  action="store_true", help="Show 3D shock plot after analysis.")
    ctrl.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    # Validate: need either a RAMSES path or a pre-built octree
    if not args.load and not args.load_octree and not args.data_path:
        parser.error("data_path is required unless --load-octree or --load is given")

    # Resolve output dir and run name
    if args.data_path:
        data_path = os.path.abspath(args.data_path)
    elif args.load_octree:
        data_path = os.path.dirname(os.path.abspath(args.load_octree))
    else:
        data_path = os.getcwd()

    if args.output:
        results_path = args.output
    else:
        results_path = os.path.join(os.path.dirname(data_path), "shockfind_results")

    if args.name:
        run_name = args.name
    elif args.data_path:
        run_name = "octree_" + os.path.basename(data_path.rstrip("/"))
    else:
        run_name = "octree_" + os.path.splitext(os.path.basename(args.load_octree))[0]

    os.makedirs(results_path, exist_ok=True)

    shock_finder = _import_shock_finder()

    if args.load:
        sf = shock_finder(name=run_name)
        sf.load_results(path=results_path, name=run_name)
        print(f"Loaded results from {results_path}/{run_name}_result.pk")
        handle = None
    else:
        handle, sf = run_octree_pipeline(
            args.data_path,
            name=run_name,
            interface=args.interface,
            octree_save_path=args.save_octree,
            octree_load_path=args.load_octree,
            max_depth=args.max_depth,
            min_depth=args.min_depth,
            max_members=args.max_members,
            max_nodes=args.max_nodes,
            vshock_min=args.vshock_min,
            rhomean=args.rhomean,
            dx=args.dx,
            N_cells=args.N_cells,
            compress=args.compress,
            div_threshold=args.div_threshold,
            grad_threshold=args.grad_threshold,
            mach_min=args.mach_min,
            vmag_min=args.vmag_min,
            use_gradTRho=not args.no_gradTRho,
            hydro_only=args.hydro_only,
            gamma=args.gamma,
            periodic=[not args.no_periodic] * 3,
            method_norm=args.method_norm,
            method_plane=args.method_plane,
            Rgrad=args.Rgrad,
            Rcylinder=args.Rcylinder,
            line_range=args.line_range,
            shock_ratio=args.shock_ratio,
            rescale=args.rescale,
            quiet=args.quiet,
        )
        sf.save_results(path=results_path, name=run_name)
        print(f"Results saved to {results_path}/{run_name}_result.pk")

    # Summary
    shocks  = sf.shocks
    flags   = np.asarray(shocks[16])
    families = np.asarray(shocks[6])
    total   = len(flags)
    ok      = int((flags == 0).sum())
    fast    = int((families == 12).sum())
    slow    = int((families == 34).sum())
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
