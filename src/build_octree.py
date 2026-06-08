"""build_octree.py — Dataset-agnostic octree builder.

Load any supported simulation snapshot through the selected interface and
produce a ShockFind-compatible OctreeHandle.  Optionally save to HDF5 for
re-use across multiple shock-finding runs.

Works as a standalone script (save to HDF5) or as a library function that
returns an OctreeHandle ready for the shockfind pipeline.

Usage (standalone):
    python src/build_octree.py ramses output_00021/ --save-octree oct.h5
    python src/build_octree.py arepo  snapshot_082.hdf5 --save-octree oct.h5

Usage (library):
    from src.build_octree import build_octree
    handle, field_names = build_octree(
        "output_00021", interface="ramses",
        max_depth=14, hydro_only=False, save_path="oct.h5")
"""
from __future__ import annotations

import os
import sys
import tempfile


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


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_octree(
    data_path: str,
    interface: str,
    *,
    save_path: str | None = None,
    # sub-box
    box=None,
    box_units: str = "code",
    # physics
    hydro_only: bool = False,
    # particle-code options (ignored for AMR)
    part_type:   int   = 0,
    hsml_factor: float = 1.0,
    hsml_geom:   str   = "sphere",
    split_mass:  bool  = False,
    # octree build — None means use the interface's code-specific default
    max_depth:   int | None = None,
    min_depth:   int | None = None,
    max_members: int | None = None,
    max_nodes:   int | None = None,
):
    """
    Build an OctreeHandle from any simulation snapshot.

    Selects the loader via ``interface`` ('ramses', 'arepo', …).
    If ``save_path`` is given the octree is serialised to HDF5; this file can
    be passed to run_octree_pipeline(load_octree=...) to skip rebuilding.

    HSML-based codes (Arepo): the octree is built via octree3d with HSML
    deposition, then converted to an OctreeHandle through a temporary HDF5
    file (or ``save_path`` if provided, saving one extra write).

    Returns
    -------
    handle      : shockfindCore_octave.OctreeHandle
    field_names : list[str]
    """
    from src.interfaces import get_interface, _import_octave

    # Build interface — only pass overrides; let the config apply its defaults
    kwargs: dict = dict(
        box=box, box_units=box_units, hydro_only=hydro_only,
        part_type=part_type, hsml_factor=hsml_factor,
        hsml_geom=hsml_geom, split_mass=split_mass,
    )
    if max_depth   is not None: kwargs["max_depth"]   = max_depth
    if min_depth   is not None: kwargs["min_depth"]   = min_depth
    if max_members is not None: kwargs["max_members"] = max_members
    if max_nodes   is not None: kwargs["max_nodes"]   = max_nodes

    iface = get_interface(interface, data_path, **kwargs)
    data  = iface.load()
    sco   = _import_shockfind_octave()

    if iface.uses_hsml:
        # ── SPH / moving-mesh: HSML deposition via octree3d ────────────────
        oc   = _import_octave()
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
        print(f"Built: {tree.num_leaves():,} leaves  {tree.num_nodes():,} nodes"
              f"  (max_depth={iface.max_depth})")

        # Convert to OctreeHandle via HDF5
        if save_path is not None:
            hdf_path = save_path
            _cleanup = False
        else:
            hdf_path = tempfile.mktemp(suffix="_octave_tmp.h5")
            _cleanup = True

        tree.write_all_hdf5(hdf_path, data.field_names)
        handle = sco.load_octree(hdf_path)

        if _cleanup:
            try:
                os.unlink(hdf_path)
            except OSError:
                pass

    else:
        # ── AMR / grid: direct build via shockfindCore_octave ──────────────
        print(f"Building octree from {len(data.positions):,} cells …")
        handle = sco.build_octree(
            data.positions, data.attrs, data.field_names,
            **iface.octree_params,
        )
        print(f"Built: {handle.num_leaves:,} leaves  {handle.num_nodes:,} nodes"
              f"  (dx = 1/{1 << handle.max_depth} = {handle.cell_size:.3e})")

        if save_path is not None:
            sco.save_octree(handle, save_path)

    if save_path is not None:
        print(f"Octree saved to {save_path}")

    return handle, data.field_names


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Build an Octave octree from any supported simulation code and save to HDF5.\n\n"
            "Examples:\n"
            "  python src/build_octree.py ramses output_00021/     --save-octree out21.h5\n"
            "  python src/build_octree.py arepo  snapshot_082.hdf5 --save-octree snap82.h5"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("interface", choices=["arepo", "ramses"],
        help="Simulation code / data format.")
    parser.add_argument("data_path",
        help="Snapshot file (Arepo HDF5) or output directory (RAMSES).")
    parser.add_argument("--save-octree", default=None, metavar="PATH",
        help="HDF5 output path (default: <stem>_octree.h5).")

    og = parser.add_argument_group("octree")
    og.add_argument("--max-depth",   type=int,   default=None,
        help="Finest level (default: 12 Arepo / 14 RAMSES).")
    og.add_argument("--min-depth",   type=int,   default=None,
        help="Coarsest pre-built level (default: 2 Arepo / 3 RAMSES).")
    og.add_argument("--max-members", type=int,   default=None,
        help="Max points per leaf (default: 8 Arepo / 1 RAMSES).")
    og.add_argument("--max-nodes",   type=int,   default=None)

    pg = parser.add_argument_group("physics")
    pg.add_argument("--hydro-only", action="store_true",
        help="Skip magnetic field loading (no B field).")

    hg = parser.add_argument_group("HSML (Arepo / SPH)")
    hg.add_argument("--part-type",   type=int,   default=0)
    hg.add_argument("--hsml-factor", type=float, default=1.0)
    hg.add_argument("--hsml-geom",   default="sphere", choices=["sphere", "cube"])
    hg.add_argument("--split-mass",  action="store_true")

    bg = parser.add_argument_group("sub-box",
        "Two modes:\n"
        "  --box XMIN XMAX YMIN YMAX ZMIN ZMAX  (explicit corners, code units)\n"
        "  --center CX CY CZ --box-side S        (centred cube, code units)")
    bg.add_argument("--box", nargs=6, type=float,
        metavar=("XMIN","XMAX","YMIN","YMAX","ZMIN","ZMAX"), default=None)
    bg.add_argument("--center", nargs=3, type=float,
        metavar=("CX","CY","CZ"), default=None)
    bg.add_argument("--box-side", type=float, default=None)

    args = parser.parse_args()

    if args.box is not None and (args.center is not None or args.box_side is not None):
        parser.error("--box and --center/--box-side are mutually exclusive")

    from src.interfaces import OctaveInterface
    box, box_units = OctaveInterface.parse_box(
        box=args.box, center=args.center, box_side=args.box_side)
    if box is not None:
        print(f"Sub-box (code units): "
              f"x=[{box[0]:.4g},{box[1]:.4g}]  "
              f"y=[{box[2]:.4g},{box[3]:.4g}]  "
              f"z=[{box[4]:.4g},{box[5]:.4g}]")

    save_path = args.save_octree
    if save_path is None:
        stem = os.path.splitext(os.path.basename(args.data_path.rstrip("/")))[0]
        save_path = f"{stem}_octree.h5"
        print(f"--save-octree not set; saving to {save_path}")

    build_octree(
        args.data_path,
        args.interface,
        save_path    = save_path,
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
    )


if __name__ == "__main__":
    main()
