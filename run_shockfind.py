#!/usr/bin/env python
"""Run the AMR-octree shock finder on one RAMSES snapshot.

    python run_shockfind.py /path/to/output_00801.h5        # aggregated snapshot (ramses_h5)
    python run_shockfind.py /path/to/output_00801           # classic output directory
    python run_shockfind.py SNAP --save-octree DIR --vshock-min 1e6 ...   # any pipeline option

Results: <name>_result.pk in shockfind_results/ next to the snapshot (or --output), name
octree_output_NNNNN (or --name). Every option after the snapshot is passed to
src/octree_shockfind_pipeline.py unchanged (see its --help, and GUIDE.md §3-§5).

An aggregated .h5 is read directly when the installed yt can read the ramses_h5 format (the
patched yt of ramses_h5/yt_patch). Otherwise, or with --reconstruct, the output directory is first
rebuilt byte for byte from the .h5 into --scratch with ramses_h5_aggregate.py (numpy + h5py only),
the finder runs on it, and the copy is deleted afterwards (--keep-reconstructed keeps it). That
needs as much free space in --scratch as the .h5 itself.
"""
import argparse
import importlib.util
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))


def yt_reads_aggregates():
    """True when the importable yt has the ramses_h5 reader (yt/frontends/ramses/_hdf5_store.py)."""
    if importlib.util.find_spec("yt") is None:
        return False
    return importlib.util.find_spec("yt.frontends.ramses._hdf5_store") is not None


def load_aggregator(ramses_h5_dir):
    """The ramses_h5_aggregate module: --ramses-h5, $RAMSES_H5_DIR, or ramses_h5 next to this repository."""
    candidates = [ramses_h5_dir, os.environ.get("RAMSES_H5_DIR"), os.path.join(os.path.dirname(HERE), "ramses_h5")]
    for d in candidates:
        if d and os.path.isfile(os.path.join(d, "ramses_h5_aggregate.py")):
            spec = importlib.util.spec_from_file_location("ramses_h5_aggregate", os.path.join(d, "ramses_h5_aggregate.py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    sys.exit("run_shockfind: cannot reconstruct the .h5: ramses_h5_aggregate.py not found "
             f"(tried {[c for c in candidates if c]}); pass --ramses-h5 DIR or set RAMSES_H5_DIR")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     epilog="Any other option goes to octree_shockfind_pipeline.py.")
    parser.add_argument("snapshot", help="output_NNNNN directory, its info_NNNNN.txt, or an aggregated output_NNNNN.h5")
    parser.add_argument("--reconstruct", action="store_true",
                        help="rebuild the output directory from the .h5 even if yt could read it directly")
    parser.add_argument("--scratch", default=None,
                        help="where to rebuild the directory (default: $TMPDIR, else the snapshot's directory)")
    parser.add_argument("--keep-reconstructed", action="store_true", help="keep the rebuilt directory")
    parser.add_argument("--ramses-h5", default=None, help="directory holding ramses_h5_aggregate.py")
    args, rest = parser.parse_known_args(argv)

    snap = os.path.abspath(args.snapshot.rstrip("/"))
    if not os.path.exists(snap):
        parser.error(f"no such snapshot: {snap}")
    is_h5 = snap.endswith(".h5")
    name = "octree_" + os.path.basename(snap).removesuffix(".h5")
    defaults = []
    if "--output" not in rest and "-out" not in rest:
        defaults += ["--output", os.path.join(os.path.dirname(snap), "shockfind_results")]
    if "--name" not in rest and "-name" not in rest:
        defaults += ["--name", name]

    from octree_shockfind_pipeline import main as pipeline_main
    if not is_h5:
        return pipeline_main([snap] + defaults + rest)
    if not args.reconstruct and yt_reads_aggregates():
        print(f"run_shockfind: reading {snap} directly (yt with the ramses_h5 reader)")
        return pipeline_main([snap] + defaults + rest)

    aggregate = load_aggregator(args.ramses_h5)
    scratch = args.scratch or os.environ.get("TMPDIR") or os.path.dirname(snap)
    workdir = tempfile.mkdtemp(prefix="shockfind_", dir=scratch)
    out_dir = os.path.join(workdir, os.path.basename(snap).removesuffix(".h5"))
    why = "requested" if args.reconstruct else "this yt has no ramses_h5 reader"
    print(f"run_shockfind: rebuilding {out_dir} from {snap} ({why})")
    try:
        aggregate.reconstruct_ramses_output(snap, out_dir)
        return pipeline_main([out_dir] + defaults + rest)
    finally:
        if args.keep_reconstructed:
            print(f"run_shockfind: kept {out_dir}")
        else:
            shutil.rmtree(workdir)


if __name__ == "__main__":
    main()
