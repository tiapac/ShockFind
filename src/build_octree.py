"""build_octree.py — Dataset-agnostic octree builder.

Loads a simulation snapshot through the selected interface, builds an Octave
octree, and saves it to an HDF5 file suitable for ShockFind or standalone use.

Usage (standalone):
    # Arepo SPH/moving-mesh snapshot
    python src/build_octree.py arepo snapshot_082.hdf5 \\
        --max-depth 12 --save-octree snapshot_082_octree.h5

    # RAMSES AMR output
    python src/build_octree.py ramses /path/to/output_00021 \\
        --max-depth 14 --save-octree output_21_octree.h5

Usage (library):
    from src.build_octree import build_from_interface
    from src.interfaces   import get_interface

    iface = get_interface("arepo", "snap.hdf5", max_depth=12, hydro_only=False)
    tree, field_names = build_from_interface(iface, save_path="snap_octree.h5")

Workflow with ShockFind:
    # Step 1 — build the octree once (can be expensive for Arepo HSML data):
    python src/build_octree.py arepo snapshot_082.hdf5 --save-octree oct.h5

    # Step 2 — run shock finding, loading the pre-built octree:
    python src/octree_shockfind_pipeline.py --load-octree oct.h5
"""
from __future__ import annotations

import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

def _import_shockfind_octave():
    """Import shockfindCore_octave (needed for the non-HSML RAMSES build path)."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Core build function
# ─────────────────────────────────────────────────────────────────────────────

def build_from_interface(iface, *, save_path: str | None = None):
    """
    Build an octree from any OctaveInterface and optionally save to HDF5.

    For SPH / moving-mesh codes (iface.uses_hsml == True):
        Uses octree3d.Octree3D with HSML deposition.
        Saves via tree.write_all_hdf5() — readable by shockfindCore_octave.load_octree().

    For AMR / grid codes (iface.uses_hsml == False):
        Uses shockfindCore_octave.build_octree() directly.
        Saves via sco.save_octree() — readable by shockfindCore_octave.load_octree().

    Parameters
    ----------
    iface     : OctaveInterface instance (already configured)
    save_path : HDF5 output path.  When None the octree is returned but not saved.

    Returns
    -------
    tree_or_handle : octree3d.Octree3D  or  shockfindCore_octave.OctreeHandle
    field_names    : list[str]
    """
    from src.interfaces import _import_octave

    data = iface.load()

    if iface.uses_hsml:
        # ── SPH / moving-mesh: HSML deposition ────────────────────────────
        oc = _import_octave()

        tree = oc.Octree3D(
            max_depth   = iface.max_depth,
            min_depth   = iface.min_depth,
            max_nodes   = iface.max_nodes,
            max_members = iface.max_members,
        )

        iface.register_fields(tree)

        tree.enable_hsml_deposit(geom=iface.hsml_geom, split_mass=iface.split_mass)
        print(f"Depositing {len(data.positions):,} particles "
              f"(hsml_geom={iface.hsml_geom}) …")
        tree.add_points_hsml(data.positions, data.hsml, data.attrs, data.particle_ids)

        print(
            f"Built octree: {tree.num_leaves():,} leaves  {tree.num_nodes():,} nodes"
            f"  (max_depth={iface.max_depth})"
        )

        if save_path is not None:
            tree.write_all_hdf5(save_path, data.field_names)
            print(f"Octree saved to {save_path}")

        return tree, data.field_names

    else:
        # ── AMR / grid: direct build ───────────────────────────────────────
        sco = _import_shockfind_octave()

        print(f"Building octree from {len(data.positions):,} cells …")
        handle = sco.build_octree(
            data.positions, data.attrs, data.field_names,
            **iface.octree_params,
        )
        print(
            f"Built octree: {handle.num_leaves:,} leaves  {handle.num_nodes:,} nodes"
            f"  (dx = 1/{1 << handle.max_depth} = {handle.cell_size:.3e})"
        )

        if save_path is not None:
            sco.save_octree(handle, save_path)
            print(f"Octree saved to {save_path}")

        return handle, data.field_names


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Build an Octave octree from any supported simulation code.\n\n"
            "Examples:\n"
            "  python src/build_octree.py arepo snap.hdf5 --save-octree snap_oct.h5\n"
            "  python src/build_octree.py ramses output_00021/ --save-octree out21_oct.h5"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "interface", choices=["arepo", "ramses"],
        help="Simulation code / data format to load.")
    parser.add_argument(
        "data_path",
        help="Snapshot file (Arepo HDF5) or output directory (RAMSES).")
    parser.add_argument(
        "--save-octree", default=None, metavar="PATH",
        help="Save octree to this HDF5 file.  "
             "Default: <data_path_stem>_octree.h5")

    # ── octree parameters ─────────────────────────────────────────────────────
    og = parser.add_argument_group("octree")
    og.add_argument("--max-depth",   type=int,   default=None,
        help="Finest octree level (default: 12 for Arepo, 14 for RAMSES).")
    og.add_argument("--min-depth",   type=int,   default=None,
        help="Coarsest pre-built level (default: 2 for Arepo, 3 for RAMSES).")
    og.add_argument("--max-members", type=int,   default=None,
        help="Max data points per leaf (default: 8 for Arepo, 1 for RAMSES).")
    og.add_argument("--max-nodes",   type=int,   default=None)

    # ── physics ───────────────────────────────────────────────────────────────
    pg = parser.add_argument_group("physics")
    pg.add_argument("--hydro-only", action="store_true",
        help="Skip magnetic field loading (hydro-only run, no B field).")

    # ── HSML (Arepo / SPH) ────────────────────────────────────────────────────
    hg = parser.add_argument_group("HSML — Arepo / SPH codes")
    hg.add_argument("--part-type",   type=int,   default=0,
        help="Arepo particle type to load (0 = gas).")
    hg.add_argument("--hsml-factor", type=float, default=1.0,
        help="Scale factor applied to stored/computed HSML.")
    hg.add_argument("--hsml-geom",   default="sphere", choices=["sphere", "cube"],
        help="HSML influence region geometry.")
    hg.add_argument("--split-mass",  action="store_true",
        help="Divide mass evenly among intersecting leaves (mass-conserving).")

    # ── sub-box ───────────────────────────────────────────────────────────────
    bg = parser.add_argument_group(
        "sub-box",
        "Two modes (mutually exclusive):\n"
        "  1) --box XMIN XMAX YMIN YMAX ZMIN ZMAX  — explicit corners in code units\n"
        "  2) --center CX CY CZ --box-side S        — cube centred at (CX,CY,CZ)")
    bg.add_argument("--box", nargs=6, type=float,
        metavar=("XMIN","XMAX","YMIN","YMAX","ZMIN","ZMAX"), default=None)
    bg.add_argument("--center", nargs=3, type=float,
        metavar=("CX","CY","CZ"), default=None)
    bg.add_argument("--box-side", type=float, default=None)

    args = parser.parse_args()

    if args.box is not None and (args.center is not None or args.box_side is not None):
        parser.error("--box and --center/--box-side are mutually exclusive")

    # Resolve sub-box
    from src.interfaces import OctaveInterface
    box, box_units = OctaveInterface.parse_box(
        box=args.box, center=args.center, box_side=args.box_side)
    if box is not None:
        print(f"Sub-box (code units): "
              f"x=[{box[0]:.4g},{box[1]:.4g}]  "
              f"y=[{box[2]:.4g},{box[3]:.4g}]  "
              f"z=[{box[4]:.4g},{box[5]:.4g}]")

    # Resolve save path
    save_path = args.save_octree
    if save_path is None:
        stem = os.path.splitext(os.path.basename(args.data_path.rstrip("/")))[0]
        save_path = f"{stem}_octree.h5"
        print(f"--save-octree not set; saving to {save_path}")

    # Build kwargs — only pass non-None overrides (let interface apply its defaults)
    kwargs: dict = dict(
        box=box,
        box_units=box_units,
        hydro_only=args.hydro_only,
        part_type=args.part_type,
        hsml_factor=args.hsml_factor,
        hsml_geom=args.hsml_geom,
        split_mass=args.split_mass,
    )
    if args.max_depth   is not None: kwargs["max_depth"]   = args.max_depth
    if args.min_depth   is not None: kwargs["min_depth"]   = args.min_depth
    if args.max_members is not None: kwargs["max_members"] = args.max_members
    if args.max_nodes   is not None: kwargs["max_nodes"]   = args.max_nodes

    from src.interfaces import get_interface
    iface = get_interface(args.interface, args.data_path, **kwargs)
    build_from_interface(iface, save_path=save_path)


if __name__ == "__main__":
    main()
