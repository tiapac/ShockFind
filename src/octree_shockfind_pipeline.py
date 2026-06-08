"""octree_shockfind_pipeline.py — ShockFind on an Octave AMR octree.

Builds the octree from any supported simulation code (via build_octree), then
runs shock candidate finding and characterisation.  All dataset-specific logic
lives in src/interfaces/; this file knows nothing about individual codes.

Usage (CLI):
    # RAMSES
    python src/octree_shockfind_pipeline.py ramses output_00021/ \\
        --save-octree oct.h5 --output results/ --name run21

    # Arepo
    python src/octree_shockfind_pipeline.py arepo snapshot_082.hdf5 \\
        --hydro-only --save-octree oct.h5 --output results/

    # Re-run shockfind on a previously built octree (skips expensive rebuild)
    python src/octree_shockfind_pipeline.py --load-octree oct.h5 \\
        --output results/ --name rerun

Usage (library):
    from src.octree_shockfind_pipeline import run_octree_pipeline
    handle, sf = run_octree_pipeline(
        "output_00021", interface="ramses",
        hydro_only=False, max_depth=14)
    sf.save_results(path="results/", name="run21")
"""
from __future__ import annotations

import os
import sys
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Internal imports
# ─────────────────────────────────────────────────────────────────────────────

def _import_shockfind_octave():
    try:
        import shockfindCore_octave
        return shockfindCore_octave
    except ImportError:
        pass
    src_dir  = os.path.dirname(os.path.abspath(__file__))
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
    src_dir        = os.path.dirname(os.path.abspath(__file__))
    shockfind_root = os.path.dirname(src_dir)
    pkg_root       = os.path.dirname(shockfind_root)
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from ShockFind import shock_finder as sf_cls
    return sf_cls


# ─────────────────────────────────────────────────────────────────────────────
# Candidate finding
# ─────────────────────────────────────────────────────────────────────────────

def find_candidates(
    sco,
    handle,
    *,
    vshock_min:    float | None = None,
    rhomean:       float | None = None,
    dx:            float | None = None,
    N_cells:       int          = 3,
    compress:      float        = 1.1,
    div_threshold: float | None = None,
    grad_threshold:float | None = None,
    mach_min:      float | None = 1.5,
    vmag_min:      float | None = None,
    use_gradTRho:  bool         = True,
    gamma:         float        = 5. / 3.,
    hydro_only:    bool         = False,
) -> list[int]:
    """Find shock candidate leaf indices.

    Physical mode (vshock_min given):
        convergence and density-gradient thresholds derived from physical quantities,
        matching shock_finder.set_thresholds().

    Raw/normalised mode (div_threshold / grad_threshold given):
        dimensionless thresholds applied directly.

    Returns list of node indices for characterise_shocks_octree().
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
    if not hydro_only:
        bx = np.asarray(handle.bx_arr)
        by = np.asarray(handle.by_arr)
        bz = np.asarray(handle.bz_arr)

    rho_safe = np.where(np.abs(rho) > 1e-30, np.abs(rho), 1e-30)

    if dx is None:
        dx = handle.cell_size

    # Convergence filter
    if vshock_min is not None:
        conv_thr = vshock_min * (compress - 1.0) / (compress * N_cells * dx)
        mask = div_v < -conv_thr
    elif div_threshold is not None:
        mask = div_v < div_threshold
    else:
        raise ValueError("Provide vshock_min or div_threshold")

    # Density gradient filter
    grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    if rhomean is not None:
        nabla_rho_thr = rhomean * (compress - 1.0) / (N_cells * dx)
        mask = mask & (grad_mag > nabla_rho_thr)
    elif grad_threshold is not None:
        mask = mask & ((grad_mag / rho_safe) > grad_threshold)

    # ∇T·∇ρ filter
    if use_gradTRho:
        T_prop = pres / rho_safe
        gTx, gTy, gTz = (np.asarray(a) for a in sco.compute_gradient_of(handle, T_prop))
        mask = mask & (gTx * gx + gTy * gy + gTz * gz > 0.0)

    # Mach filter
    if mach_min is not None and mach_min > 0.0:
        cs   = np.sqrt(gamma * np.maximum(pres, 0.0) / rho_safe)
        vmag = np.sqrt(vx**2 + vy**2 + vz**2)
        if hydro_only:
            mach = vmag / np.maximum(cs, 1e-30)
        else:
            bmag = np.sqrt(bx**2 + by**2 + bz**2)
            ca   = bmag / np.sqrt(4.0 * np.pi * rho_safe)
            mach = vmag / np.maximum(0.7 * np.minimum(cs, ca), 1e-30)
        mask = mask & (mach >= mach_min)

    # Velocity magnitude filter
    if vmag_min is not None and vmag_min > 0.0:
        vmag = np.sqrt(vx**2 + vy**2 + vz**2)
        mask = mask & (vmag >= vmag_min)

    node_indices = np.asarray(handle.leaf_node_indices)
    return [int(nid) for nid in node_indices[mask]]


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_octree_pipeline(
    data_path: str | None = None,
    interface: str = "ramses",
    name: str = "octree_run",
    *,
    # ── Octree build (forwarded to build_octree; ignored when load_octree given)
    box=None,
    box_units:   str   = "code",
    hydro_only:  bool  = False,
    part_type:   int   = 0,
    hsml_factor: float = 1.0,
    hsml_geom:   str   = "sphere",
    split_mass:  bool  = False,
    max_depth:   int | None = None,
    min_depth:   int | None = None,
    max_members: int | None = None,
    max_nodes:   int | None = None,
    # ── HDF5 persistence ─────────────────────────────────────────────────────
    save_octree: str | None = None,
    load_octree: str | None = None,
    # ── Candidate finding — physical mode ─────────────────────────────────────
    vshock_min:    float | None = None,
    rhomean:       float | None = None,
    dx:            float | None = None,
    N_cells:       int          = 3,
    compress:      float        = 1.1,
    # ── Candidate finding — raw mode ─────────────────────────────────────────
    div_threshold:  float = -0.1,
    grad_threshold: float = 0.1,
    # ── Additional candidate filters ─────────────────────────────────────────
    mach_min:     float | None = 1.5,
    vmag_min:     float | None = None,
    use_gradTRho: bool         = True,
    # ── Characterisation ─────────────────────────────────────────────────────
    gamma:        float        = 5. / 3.,
    periodic:     list | None  = None,
    method_norm:  str          = "point_gradient",
    method_plane: str          = "point_field",
    Rgrad:        int          = 3,
    Rcylinder:    int          = 3,
    line_range:   int          = 10,
    field_ref:    int          = 0,
    shock_ratio:  float        = 1.1,
    # ── Control ──────────────────────────────────────────────────────────────
    rescale: int  = 0,
    quiet:   bool = False,
):
    """
    End-to-end pipeline: dataset → octree → shock candidates → shock results.

    Interface selection is handled entirely by ``interface`` and the
    src/interfaces/ layer — no code-specific logic here.

    Parameters
    ----------
    data_path   : snapshot file or dataset directory.
                  Not required when load_octree is given.
    interface   : 'ramses' | 'arepo' | … (see src/interfaces/)
    load_octree : load a pre-built octree HDF5 and skip the build step.
    save_octree : save the built octree to this HDF5 path after building.

    Returns
    -------
    handle : OctreeHandle
    sf     : shock_finder instance with populated results
    """
    if load_octree is None and data_path is None:
        raise ValueError("Provide data_path or load_octree.")

    sco          = _import_shockfind_octave()
    shock_finder = _import_shock_finder()

    if periodic is None:
        periodic = [False, False, False]

    # ── 1. Octree ─────────────────────────────────────────────────────────────
    if load_octree is not None:
        print(f"Loading octree from {load_octree} …")
        handle = sco.load_octree(load_octree)
        print(f"  {handle.num_leaves:,} leaves  {handle.num_nodes:,} nodes  "
              f"(dx = 1/{1 << handle.max_depth} = {handle.cell_size:.3e})")
    else:
        from src.build_octree import build_octree
        handle, _ = build_octree(
            data_path, interface,
            save_path   = save_octree,
            box         = box,
            box_units   = box_units,
            hydro_only  = hydro_only,
            part_type   = part_type,
            hsml_factor = hsml_factor,
            hsml_geom   = hsml_geom,
            split_mass  = split_mass,
            max_depth   = max_depth,
            min_depth   = min_depth,
            max_members = max_members,
            max_nodes   = max_nodes,
        )

    # ── 2. Candidates ─────────────────────────────────────────────────────────
    print("Finding candidates …")
    candidates = find_candidates(
        sco, handle,
        vshock_min    = vshock_min,
        rhomean       = rhomean,
        dx            = dx,
        N_cells       = N_cells,
        compress      = compress,
        div_threshold = div_threshold,
        grad_threshold= grad_threshold,
        mach_min      = mach_min,
        vmag_min      = vmag_min,
        use_gradTRho  = use_gradTRho,
        gamma         = gamma,
        hydro_only    = hydro_only,
    )
    print(f"  {len(candidates):,} candidate leaves")

    for _ in range(rescale):
        candidates = candidates[::2]
    if rescale > 0:
        print(f"  {len(candidates):,} after rescale={rescale}")

    # ── 3. Characterise ───────────────────────────────────────────────────────
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

    # ── 4. Wrap into shock_finder ─────────────────────────────────────────────
    sf = shock_finder(name=name)
    sf.shocks = data
    sf.header = header
    sf.shocks_data(dx=handle.cell_size)

    return handle, sf


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "ShockFind on an Octave AMR octree (any simulation code).\n\n"
            "Examples:\n"
            "  python src/octree_shockfind_pipeline.py ramses output_00021/ --name run21\n"
            "  python src/octree_shockfind_pipeline.py arepo  snap.hdf5 --hydro-only\n"
            "  python src/octree_shockfind_pipeline.py --load-octree oct.h5 --name rerun"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Positional: interface + data_path (both optional when --load-octree given)
    parser.add_argument("interface", nargs="?", default=None,
        choices=["ramses", "arepo"],
        help="Simulation code / interface (omit when using --load-octree).")
    parser.add_argument("data_path", nargs="?", default=None,
        help="Snapshot / dataset path (omit when using --load-octree).")

    parser.add_argument("-out", "--output", default=None,
        help="Results directory (default: <data_path>/../shockfind_results/).")
    parser.add_argument("-name", "--name", default=None,
        help="Run name for output files.")

    # ── Octree build ─────────────────────────────────────────────────────────
    og = parser.add_argument_group("octree build")
    og.add_argument("--max-depth",   type=int,   default=None,
        help="Finest level (default: 12 Arepo / 14 RAMSES).")
    og.add_argument("--min-depth",   type=int,   default=None)
    og.add_argument("--max-members", type=int,   default=None)
    og.add_argument("--max-nodes",   type=int,   default=None)

    # ── HDF5 persistence ──────────────────────────────────────────────────────
    hdf = parser.add_argument_group("HDF5 octree persistence")
    hdf.add_argument("--save-octree", default=None, metavar="PATH",
        help="Save built octree to HDF5 (reuse with --load-octree).")
    hdf.add_argument("--load-octree", default=None, metavar="PATH",
        help="Load pre-built octree HDF5; skips data loading and tree build.")

    # ── Physics ───────────────────────────────────────────────────────────────
    pg = parser.add_argument_group("physics")
    pg.add_argument("--hydro-only", action="store_true",
        help="Hydro-only mode: skip B field, sonic shock classification (family=1).")

    # ── HSML (Arepo / SPH) ────────────────────────────────────────────────────
    hg = parser.add_argument_group("HSML (Arepo / SPH codes)")
    hg.add_argument("--part-type",   type=int,   default=0)
    hg.add_argument("--hsml-factor", type=float, default=1.0)
    hg.add_argument("--hsml-geom",   default="sphere", choices=["sphere", "cube"])
    hg.add_argument("--split-mass",  action="store_true")

    # ── Sub-box ───────────────────────────────────────────────────────────────
    bg = parser.add_argument_group("sub-box",
        "Two modes:\n"
        "  --box XMIN XMAX YMIN YMAX ZMIN ZMAX  (explicit corners)\n"
        "  --center CX CY CZ --box-side S        (centred cube)")
    bg.add_argument("--box", nargs=6, type=float,
        metavar=("XMIN","XMAX","YMIN","YMAX","ZMIN","ZMAX"), default=None)
    bg.add_argument("--center", nargs=3, type=float,
        metavar=("CX","CY","CZ"), default=None)
    bg.add_argument("--box-side", type=float, default=None)

    # ── Candidate thresholds — physical mode ──────────────────────────────────
    cand = parser.add_argument_group("candidate thresholds — physical mode",
        "Mirrors shock_finder.set_thresholds(). Provide vshock-min + rhomean to use "
        "physical thresholds; otherwise raw dimensionless thresholds are used.")
    cand.add_argument("--vshock-min", type=float, default=1e5)
    cand.add_argument("--rhomean",    type=float, default=None)
    cand.add_argument("--dx",         type=float, default=None)
    cand.add_argument("--N-cells",    type=int,   default=3)
    cand.add_argument("--compress",   type=float, default=1.1)

    # ── Candidate thresholds — raw mode ───────────────────────────────────────
    raw = parser.add_argument_group("candidate thresholds — raw/normalised mode")
    raw.add_argument("--div-threshold",  type=float, default=-0.1)
    raw.add_argument("--grad-threshold", type=float, default=0.1)

    # ── Additional candidate filters ──────────────────────────────────────────
    filt = parser.add_argument_group("candidate filters")
    filt.add_argument("--mach-min", type=float, default=1.5)
    filt.add_argument("--vmag-min", type=float, default=None)
    filt.add_argument("--no-gradTRho", action="store_true",
        help="Disable ∇T·∇ρ > 0 filter.")

    # ── Characterisation ──────────────────────────────────────────────────────
    char = parser.add_argument_group("characterisation")
    char.add_argument("--gamma",        type=float, default=1.666667)
    char.add_argument("--no-periodic",  action="store_true")
    char.add_argument("--method-norm",  default="point_gradient",
        choices=["point_gradient", "average_gradient"])
    char.add_argument("--method-plane", default="point_field",
        choices=["point_field", "average_field"])
    char.add_argument("--Rgrad",        type=int,   default=3)
    char.add_argument("--Rcylinder",    type=int,   default=3)
    char.add_argument("--line-range",   type=int,   default=10)
    char.add_argument("--shock-ratio",  type=float, default=1.1)

    # ── Control ───────────────────────────────────────────────────────────────
    ctrl = parser.add_argument_group("control")
    ctrl.add_argument("--rescale", type=int, default=0)
    ctrl.add_argument("--plot",    action="store_true")
    ctrl.add_argument("--quiet",   action="store_true")

    args = parser.parse_args()

    # Validate
    if args.load_octree is None and (args.interface is None or args.data_path is None):
        parser.error("interface and data_path are required unless --load-octree is given")
    if args.box is not None and (args.center is not None or args.box_side is not None):
        parser.error("--box and --center/--box-side are mutually exclusive")

    # Sub-box
    from src.interfaces import OctaveInterface
    box, box_units = OctaveInterface.parse_box(
        box=args.box, center=args.center, box_side=args.box_side)

    # Resolve output dir and run name
    if args.data_path:
        data_path = os.path.abspath(args.data_path)
    else:
        data_path = os.path.dirname(os.path.abspath(args.load_octree))

    results_path = args.output or os.path.join(
        os.path.dirname(data_path), "shockfind_results")
    os.makedirs(results_path, exist_ok=True)

    if args.name:
        run_name = args.name
    elif args.data_path:
        run_name = "octree_" + os.path.basename(args.data_path.rstrip("/"))
    else:
        run_name = "octree_" + os.path.splitext(
            os.path.basename(args.load_octree))[0]

    _, sf = run_octree_pipeline(
        args.data_path,
        args.interface or "ramses",
        name         = run_name,
        box          = box,
        box_units    = box_units,
        hydro_only   = args.hydro_only,
        part_type    = args.part_type,
        hsml_factor  = args.hsml_factor,
        hsml_geom    = args.hsml_geom,
        split_mass   = args.split_mass,
        max_depth    = args.max_depth,
        min_depth    = args.min_depth,
        max_members  = args.max_members,
        max_nodes    = args.max_nodes,
        save_octree  = args.save_octree,
        load_octree  = args.load_octree,
        vshock_min   = args.vshock_min,
        rhomean      = args.rhomean,
        dx           = args.dx,
        N_cells      = args.N_cells,
        compress     = args.compress,
        div_threshold  = args.div_threshold,
        grad_threshold = args.grad_threshold,
        mach_min     = args.mach_min,
        vmag_min     = args.vmag_min,
        use_gradTRho = not args.no_gradTRho,
        gamma        = args.gamma,
        periodic     = [not args.no_periodic] * 3,
        method_norm  = args.method_norm,
        method_plane = args.method_plane,
        Rgrad        = args.Rgrad,
        Rcylinder    = args.Rcylinder,
        line_range   = args.line_range,
        shock_ratio  = args.shock_ratio,
        rescale      = args.rescale,
        quiet        = args.quiet,
    )

    sf.save_results(path=results_path, name=run_name)
    print(f"Results saved to {results_path}/{run_name}_result.pk")

    shocks   = sf.shocks
    flags    = np.asarray(shocks[16])
    families = np.asarray(shocks[6])
    total    = len(flags)
    ok       = int((flags == 0).sum())
    fast     = int((families == 12).sum())
    slow     = int((families == 34).sum())
    sonic    = int((families == 1).sum())
    print(f"\nTotal: {total}  |  OK (flag=0): {ok}  |  "
          f"Fast: {fast}  Slow: {slow}  Sonic: {sonic}")

    if args.plot:
        import matplotlib.pyplot as plt
        ax, fig = sf.plot3D(types="fs")
        fig_path = os.path.join(results_path, f"{run_name}_3Dplot.png")
        fig.savefig(fig_path, dpi=150)
        print(f"3D plot saved to {fig_path}")
        plt.show()


if __name__ == "__main__":
    main()
